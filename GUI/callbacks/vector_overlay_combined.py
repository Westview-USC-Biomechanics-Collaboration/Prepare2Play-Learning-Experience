import os
import pandas as pd
import numpy as np
import cv2
from vector_overlay.vectoroverlay_GUI import VectorOverlay
from GUI.callbacks.ledSyncing_with_detection_system import new_led
from GUI.callbacks import global_variable
from Util.force_boundary_finder import find_force_boundaries, get_trimmed_subset
from vector_overlay.com_processor_modified import BoundaryProcessor as Processor
import warnings
import tkinter as tk
from tkinter import messagebox
from GUI.callbacks.manual_alignment import AlignmentGUI

warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated.*")

MAX_COM_WORKERS  = 6
FORCE_THRESHOLD  = 50
BOUNDARY_PADDING = 10
SHOW_LANDMARKS   = False
USE_DETECTION_SYSTEM = True

FORCE_HZ = 1200


def _apply_alignment(force_df, force_align, video_align, step_size):
    """
    Replicate alignCallback logic exactly:

        offset = force_align - video_align
        if offset > 0:  trim front of force data by offset * step_size rows
        if offset < 0:  pad front with NaN rows

    Then assign FrameNumber = 0, 1, 2, ... so it maps 1:1 with video frames
    starting from frame 0 (video is also seeked to 0 after trimming).

    Parameters
    ----------
    force_df     : raw self.Force.data DataFrame
    force_align  : float — force time (s) the user marked  (app.force_align)
    video_align  : int   — video frame the user marked      (app.video_align)
    step_size    : int   — force rows per video frame = round(FORCE_HZ / video_fps)
    """
    df = force_df.copy()

    # Rename columns
    rename = {
        'abs time (s)': 'Time(s)',
        'Fx':     'FP1_Fx',  'Fy':     'FP1_Fy',  'Fz':     'FP1_Fz',
        '|Ft|':   'FP1_|F|', 'Ax':     'FP1_Ax',  'Ay':     'FP1_Ay',
        'Fx.1':   'FP2_Fx',  'Fy.1':   'FP2_Fy',  'Fz.1':   'FP2_Fz',
        '|Ft|.1': 'FP2_|F|', 'Ax.1':   'FP2_Ax',  'Ay.1':   'FP2_Ay',
        'Fx.2':   'FP3_Fx',  'Fy.2':   'FP3_Fy',  'Fz.2':   'FP3_Fz',
        '|Ft|.2': 'FP3_|F|', 'Ax.2':   'FP3_Ax',  'Ay.2':   'FP3_Ay',
        'Fx1': 'FP1_Fx', 'Fy1': 'FP1_Fy', 'Fz1': 'FP1_Fz', '|Ft1|': 'FP1_|F|',
        'Ax1': 'FP1_Ax', 'Ay1': 'FP1_Ay',
        'Fx2': 'FP2_Fx', 'Fy2': 'FP2_Fy', 'Fz2': 'FP2_Fz', '|Ft2|': 'FP2_|F|',
        'Ax2': 'FP2_Ax', 'Ay2': 'FP2_Ay',
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    # Exact alignCallback logic
    offset = force_align - video_align
    print(f"[align] force_align={force_align}, video_align={video_align}, "
          f"offset={offset}, step_size={step_size}")

    if offset > 0:
        # Force starts later than video — trim front of force
        rows_to_drop = int(offset * step_size)
        df = df.iloc[rows_to_drop:].reset_index(drop=True)
        print(f"[align] Trimmed {rows_to_drop} rows from front of force data")
    elif offset < 0:
        # Force starts earlier than video — pad front with NaN
        rows_to_add = int(-offset * step_size)
        nan_rows = pd.DataFrame(np.nan, index=range(rows_to_add), columns=df.columns)
        df = pd.concat([nan_rows, df], ignore_index=True)
        print(f"[align] Padded {rows_to_add} NaN rows at front of force data")

    # Subsample to 1 row per video frame
    df = df.iloc[::step_size].reset_index(drop=True)

    # FrameNumbers start at 0 — they index into the trimmed/output video,
    # not the original. The video is seeked to video_start_frame separately.
    df['FrameNumber'] = range(len(df))

    if 'FP3_Fz' in df.columns:
        df['FP_LED_Signal'] = np.sign(df['FP3_Fz'].fillna(0))

    print(f"[align] df rows={len(df)}, FrameNumbers 0-{len(df)-1}")
    return df


def vectorOverlayWithAlignmentCallback(self, video, view, num):

    print(f"[DEBUG] View: {view}")
    print("[INFO] Starting Vector Overlay with Alignment...")
    parent_path = os.path.dirname(video.path)
    video_file  = os.path.basename(video.path)
    force_file  = os.path.basename(self.Force.path)

    # Write first frame for corner detection
    cap = cv2.VideoCapture(video.path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    ret, frame = cap.read()
    cv2.imwrite(os.path.join(parent_path, "First_Frame_of_Video.PNG"), frame)
    cap.release()

    # ======================================================================
    # STEP 1: ALIGNMENT
    # ======================================================================
    print("\n[STEP 1] Aligning video and force data...")

    auto_succeeded  = False
    auto_lag        = 0
    df_aligned_auto = None
    use_manual      = False

    try:
        auto_lag, df_aligned_auto = new_led(
            self, view, parent_path, video_file, force_file,
            use_detection_system=USE_DETECTION_SYSTEM
        )
        auto_succeeded = True
        print(f"[INFO] LED auto-alignment succeeded. Lag: {auto_lag} frames")
    except Exception as e:
        print(f"[WARNING] LED auto-alignment failed: {e}")

    if auto_succeeded:
        use_manual = messagebox.askyesno(
            "Alignment",
            f"Auto-alignment detected a lag of {auto_lag} frames.\n\n"
            "Open manual alignment to verify or override?",
            parent=self.master
        )
    else:
        messagebox.showwarning(
            "Auto-Alignment Failed",
            "LED auto-alignment could not determine the lag.\n"
            "Please align manually using the next window.",
            parent=self.master
        )
        use_manual = True

    if use_manual:
        print("[INFO] Opening manual alignment window...")
        alignment_root = tk.Toplevel(self.master)
        app = AlignmentGUI(alignment_root, video, self.Force)
        self.master.wait_window(alignment_root)

        video_fps  = app.video_fps if app.video_fps else 59
        step_size  = round(FORCE_HZ / video_fps)  # force rows per video frame

        # Both are FRAME NUMBERS — exact same units as original alignCallback.
        # offset = force_align - video_align  (frames)
        # rows_to_drop = offset * step_size   (force rows)
        force_align       = app.force_align   # frame number where force was marked
        video_align       = app.video_align   # frame number where video was marked
        print(f"[INFO] video_fps={video_fps}, step_size={step_size}")
        print(f"[INFO] video_align={video_align} frames, force_align={force_align} frames")
        print(f"[INFO] offset={force_align - video_align} frames")

        df_aligned = _apply_alignment(
            self.Force.data.copy(),
            force_align=force_align,
            video_align=video_align,
            step_size=step_size
        )

        # Apply plate swap for this view if needed
        try:
            from GUI.callbacks.led_detection_system import config_map
            config = config_map[view]()
            if config.plate_swap:
                print(f"[INFO] Applying plate swap for {view}")
                for c1, c2 in [('FP1_Fx','FP2_Fx'),('FP1_Fy','FP2_Fy'),('FP1_Fz','FP2_Fz'),
                                ('FP1_|F|','FP2_|F|'),('FP1_Ax','FP2_Ax'),('FP1_Ay','FP2_Ay')]:
                    if c1 in df_aligned.columns and c2 in df_aligned.columns:
                        df_aligned[c1], df_aligned[c2] = df_aligned[c2].copy(), df_aligned[c1].copy()
        except Exception as e:
            print(f"[WARNING] Plate swap check failed: {e}")

        lag = 0  # video is seeked to frame 0 since force is trimmed to match

    else:
        df_aligned = df_aligned_auto
        lag = auto_lag

    # Column rename guard
    col_rename = {
        'Fx1': 'FP1_Fx', 'Fy1': 'FP1_Fy', 'Fz1': 'FP1_Fz', 'Ft1': 'FP1_|F|',
        'Ax1': 'FP1_Ax', 'Ay1': 'FP1_Ay',
        'Fx2': 'FP2_Fx', 'Fy2': 'FP2_Fy', 'Fz2': 'FP2_Fz', 'Ft2': 'FP2_|F|',
        'Ax2': 'FP2_Ax', 'Ay2': 'FP2_Ay',
        'abs time (s)': 'Time(s)'
    }
    df_aligned.rename(columns={k: v for k, v in col_rename.items()
                                if k in df_aligned.columns}, inplace=True)

    self.state.df_aligned = df_aligned
    self.state.global_lag = lag

    print(f"[INFO] df_aligned: {df_aligned.shape}, cols: {list(df_aligned.columns)}")
    print(f"[INFO] FrameNumbers: {df_aligned['FrameNumber'].min()}-{df_aligned['FrameNumber'].max()}")

    # ======================================================================
    # STEP 2: FORCE BOUNDARIES
    # ======================================================================
    print("\n[STEP 2] Finding force boundaries...")
    try:
        boundary_start, boundary_end = find_force_boundaries(
            df_aligned, threshold=FORCE_THRESHOLD, padding_frames=BOUNDARY_PADDING
        )
        self.state.boundary_start = boundary_start
        self.state.boundary_end   = boundary_end
        print(f"[INFO] Boundaries: {boundary_start}-{boundary_end}")
    except Exception as e:
        print(f"[ERROR] Boundary detection failed: {e}")
        boundary_start = int(df_aligned['FrameNumber'].min())
        boundary_end   = int(df_aligned['FrameNumber'].max())
        self.state.boundary_start = boundary_start
        self.state.boundary_end   = boundary_end

    # ======================================================================
    # STEP 3: COM CALCULATION
    # ======================================================================
    com_csv_path = None
    if view != "Top View":
        print("\n[STEP 3] Running COM calculation...")
        com_csv_path = os.path.join(parent_path, "pose_landmarks.csv")
        try:
            sex = global_variable.globalVariable.sex or 'm'
            processor = Processor(video.path)
            processor.SaveToTxt(
                sex=sex,
                filename=com_csv_path,
                confidencelevel=0.85,
                displayCOM=True,
                start_frame=boundary_start,
                end_frame=boundary_end,
                max_workers=MAX_COM_WORKERS
            )
            if os.path.exists(com_csv_path):
                from Util.COM_helper import COM_helper
                self.COM_helper = COM_helper(com_csv_path)
                print(f"[INFO] COM updated: {com_csv_path}")
        except Exception as e:
            print(f"[ERROR] COM failed: {e}")
            import traceback; traceback.print_exc()
            com_csv_path = None

    # ======================================================================
    # STEP 4: VECTOR OVERLAY
    # Video is seeked to frame 0 (manual) or lag (auto) since force data
    # has already been trimmed/padded to start in sync with the video.
    # ======================================================================
    print("\n[STEP 4] Running vector overlay...")
    df_trimmed = get_trimmed_subset(
        self.state.df_aligned, self.state.boundary_start, self.state.boundary_end
    )
    temp_video = "vector_overlay_temp.mp4"

    try:
        # For manual alignment: seek to the frame the user chose.
        # lag=0 after manual trim, but we seek to current_frame so the
        # vector overlay starts rendering from the right place in the video.
        seek_to = app.current_frame if use_manual else lag
        print(f"[INFO] Seeking video to frame {seek_to}")
        video.cam.set(cv2.CAP_PROP_POS_FRAMES, seek_to)

        v = VectorOverlay(data=df_trimmed, video=video.cam, view=view)
        v.check_corner(view)

        col_rename2 = {
            'Fx1': 'FP1_Fx', 'Fy1': 'FP1_Fy', 'Fz1': 'FP1_Fz', 'Ft1': 'FP1_|F|',
            'Ax1': 'FP1_Ax', 'Ay1': 'FP1_Ay',
            'Fx2': 'FP2_Fx', 'Fy2': 'FP2_Fy', 'Fz2': 'FP2_Fz', 'Ft2': 'FP2_|F|',
            'Ax2': 'FP2_Ax', 'Ay2': 'FP2_Ay',
            'abs time (s)': 'Time(s)'
        }
        df_trimmed.rename(columns={k: v for k, v in col_rename2.items()
                                    if k in df_trimmed.columns}, inplace=True)
        self.state.df_trimmed = df_trimmed.reset_index(drop=True)

        print("[COLUMNS] VectorOverlay:", list(df_trimmed.columns))

        overlay_kwargs = dict(
            df_aligned=df_trimmed,
            outputName=temp_video,
            lag=lag,
            com_csv_path=com_csv_path,
            show_landmarks=SHOW_LANDMARKS,
            boundary_start=self.state.boundary_start,
            boundary_end=self.state.boundary_end
        )

        frames = []
        if view == "Long View":
            frames = v.LongVectorOverlay(**overlay_kwargs)
        elif view == "Side1 View":
            frames = v.SideVectorOverlay(**overlay_kwargs, is_side1=True)
        elif view == "Side2 View":
            frames = v.SideVectorOverlay(**overlay_kwargs, is_side1=False)
        elif view == "Top View":
            frames = v.TopVectorOverlay(**{**overlay_kwargs, 'com_csv_path': None})

        print(f"[INFO] Vector overlay complete: {temp_video}")

        video.vector_cam = cv2.VideoCapture(temp_video)
        video.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
        video.vector_cam.set(cv2.CAP_PROP_POS_FRAMES, self.state.loc)

        self.canvasManager.photo_image3 = self.frameConverter.cvToPillow(camera=video.vector_cam)
        self.canvasManager.canvas3.create_image(200, 150,
                                                image=self.canvasManager.photo_image3,
                                                anchor="center")
        self.state.vector_overlay_enabled = True
        print("[SUCCESS] All processing complete!")
        return frames

    except Exception as e:
        print(f"[ERROR] Vector overlay failed: {e}")
        import traceback; traceback.print_exc()