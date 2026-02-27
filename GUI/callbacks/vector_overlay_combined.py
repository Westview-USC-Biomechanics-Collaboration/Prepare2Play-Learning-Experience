import threading
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
from tkinter import ttk, filedialog, messagebox
from GUI.callbacks.manual_alignment import AlignmentGUI

# Suppress specific deprecation warnings
warnings.filterwarnings(
    "ignore", 
    message="SymbolDatabase.GetPrototype() is deprecated.*"
)

# Configurable settings
MAX_COM_WORKERS = 6  # Easily adjustable number of workers for COM calculation
FORCE_THRESHOLD = 50  # Minimum force in Newtons to include in processing
BOUNDARY_PADDING = 10  # Extra frames before/after force threshold
SHOW_LANDMARKS = False  # Show green landmark dots (set to True to enable)
USE_DETECTION_SYSTEM = True  # Use new LED detection system (set to False for original method)

# Fixed rates — video is always 120 fps, force is always 1200 Hz
VIDEO_FPS    = 120
FORCE_HZ     = 1200
FORCE_STEP   = FORCE_HZ // VIDEO_FPS   # = 10  (subsample step matching new_led's iloc[::10])
# After subsampling, force has VIDEO_FPS rows/second, so 1 force row = 1 video frame.
# Therefore: lag in video frames = round(offset_seconds * VIDEO_FPS)
#            FrameNumber = lag + row_index  (1:1 with video frames)


def _build_df_aligned_from_manual(force_data, lag_video_frames):
    """
    Reproduce the same df_aligned structure that new_led produces,
    using a manually specified lag (in VIDEO FRAMES) instead of cross-correlation.

    new_led subsamples force at ::10 (1200 Hz → 120 Hz = video fps),
    then sets FrameNumber = range(lag, lag + len(subset)).
    After subsampling, 1 force row = 1 video frame, so FrameNumbers map 1:1.

    Parameters
    ----------
    force_data        : raw self.Force.data DataFrame (already numeric)
    lag_video_frames  : int — video frame index where force t=0 should land
                        = round(offset_seconds * VIDEO_FPS)
    """
    df = force_data.copy()

    # ── Rename columns to match new_led output ────────────────────────────
    force_dict = {
        'abs time (s)': 'Time(s)',
        'Fx':     'FP1_Fx',  'Fy':     'FP1_Fy',  'Fz':     'FP1_Fz',
        '|Ft|':   'FP1_|F|', 'Ax':     'FP1_Ax',  'Ay':     'FP1_Ay',
        'Fx.1':   'FP2_Fx',  'Fy.1':   'FP2_Fy',  'Fz.1':   'FP2_Fz',
        '|Ft|.1': 'FP2_|F|', 'Ax.1':   'FP2_Ax',  'Ay.1':   'FP2_Ay',
        'Fx.2':   'FP3_Fx',  'Fy.2':   'FP3_Fy',  'Fz.2':   'FP3_Fz',
        '|Ft|.2': 'FP3_|F|', 'Ax.2':   'FP3_Ax',  'Ay.2':   'FP3_Ay',
    }
    rename_map = {k: v for k, v in force_dict.items() if k in df.columns}
    df.rename(columns=rename_map, inplace=True)

    # Also handle Fx1/Fz1 style names (your app's fileReader output)
    force_dict2 = {
        'Fx1': 'FP1_Fx', 'Fy1': 'FP1_Fy', 'Fz1': 'FP1_Fz', '|Ft1|': 'FP1_|F|',
        'Ax1': 'FP1_Ax', 'Ay1': 'FP1_Ay',
        'Fx2': 'FP2_Fx', 'Fy2': 'FP2_Fy', 'Fz2': 'FP2_Fz', '|Ft2|': 'FP2_|F|',
        'Ax2': 'FP2_Ax', 'Ay2': 'FP2_Ay',
    }
    rename_map2 = {k: v for k, v in force_dict2.items() if k in df.columns}
    df.rename(columns=rename_map2, inplace=True)

    # ── Subsample ×10 — matches new_led's df_force.iloc[::10] ────────────
    # 1200 Hz / 10 = 120 Hz = video fps → 1 row per video frame
    df_subset = df.iloc[::FORCE_STEP].reset_index(drop=True)

    # ── LED signal column (matches new_led) ───────────────────────────────
    if 'FP3_Fz' in df_subset.columns:
        df_subset['FP_LED_Signal'] = np.sign(df_subset['FP3_Fz'])

    # ── FrameNumber: 1 force row = 1 video frame after subsampling ────────
    # new_led: df_force_subset['FrameNumber'] = range(lag, lag + len(subset))
    n = len(df_subset)
    df_subset['FrameNumber'] = list(range(lag_video_frames, lag_video_frames + n))

    print(f"[_build_df_aligned] lag={lag_video_frames} video frames, "
          f"rows={n}, FrameNumbers {lag_video_frames}–{lag_video_frames + n - 1}")

    return df_subset


def vectorOverlayWithAlignmentCallback(self, video, view, num):

    print(f"[DEBUG] The view the vector overlay program is getting is {view}")
    print("[INFO] Starting Vector Overlay with Alignment...")
    parent_path = os.path.dirname(video.path)
    video_file = os.path.basename(video.path)
    force_file = os.path.basename(self.Force.path)

    cap = cv2.VideoCapture(video.path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    ret, frame = cap.read()
    cv2.imwrite(os.path.join(parent_path, "First_Frame_of_Video.PNG"), frame)
    cap.release()

    # ======================================================================
    # STEP 1: ALIGN VIDEO AND FORCE DATA
    # ======================================================================
    lag = 0
    df_aligned = None
    alignment_source = "none"
    print("\n[STEP 1] Aligning video and force data...")

    try:
        lag, df_aligned = new_led(
            self, view, parent_path, video_file, force_file,
            use_detection_system=USE_DETECTION_SYSTEM
        )
        alignment_source = "auto"
        print(f"[INFO] LED auto-alignment succeeded. Lag: {lag} frames")
    except Exception as e:
        print(f"[WARNING] LED auto-alignment failed: {e}")
        messagebox.showwarning(
            "Auto-Alignment Failed",
            "LED auto-alignment could not determine the lag.\n"
            "Please align manually using the next window.",
            parent=self.master
        )

    # ── Ask user whether to accept auto result or align manually ──────────
    use_manual = False
    if alignment_source == "auto":
        lag_seconds = lag / VIDEO_FPS
        answer = messagebox.askyesno(
            "Alignment",
            f"Auto-alignment detected a lag of {lag} frames ({lag_seconds:.3f}s).\n\n"
            "Open manual alignment to verify or override?",
            parent=self.master
        )
        use_manual = answer
    else:
        use_manual = True

    # ── Manual alignment popup ────────────────────────────────────────────
    if use_manual:
        print("[INFO] Opening manual alignment window...")
        alignment_root = tk.Toplevel(self.master)
        app = AlignmentGUI(alignment_root, video, self.Force)
        self.master.wait_window(alignment_root)  # blocks until Confirm & Close

        manual_offset_seconds = app.offset

        # offset_seconds × VIDEO_FPS = lag in video frames (1:1 with FrameNumber)
        lag = round(manual_offset_seconds * VIDEO_FPS)
        alignment_source = "manual"

        print(f"[INFO] Manual offset: {manual_offset_seconds:.4f}s = {lag} video frames")

        df_aligned = _build_df_aligned_from_manual(self.Force.data.copy(), lag)
        print(f"[INFO] df_aligned: {df_aligned.shape}, "
              f"FrameNumbers {df_aligned['FrameNumber'].min()}–{df_aligned['FrameNumber'].max()}")

    print(f"[INFO] Final lag: {lag} frames (source: {alignment_source})")
    print(f"Video file: {video_file}")
    print(f"Force file: {force_file}")

    # ── Column rename (guard for any remaining old-style names) ───────────
    column_rename = {
        'Fx1': 'FP1_Fx', 'Fy1': 'FP1_Fy', 'Fz1': 'FP1_Fz', 'Ft1': 'FP1_|F|',
        'Ax1': 'FP1_Ax', 'Ay1': 'FP1_Ay',
        'Fx2': 'FP2_Fx', 'Fy2': 'FP2_Fy', 'Fz2': 'FP2_Fz', 'Ft2': 'FP2_|F|',
        'Ax2': 'FP2_Ax', 'Ay2': 'FP2_Ay',
        'Fx3': 'FP3_Fx', 'Fy3': 'FP3_Fy', 'Fz3': 'FP3_Fz', 'Ft3': 'FP3_|F|',
        'Ax3': 'FP3_Ax', 'Ay3': 'FP3_Ay',
        'abs time (s)': 'Time(s)'
    }
    df_aligned.rename(columns={k: v for k, v in column_rename.items()
                                if k in df_aligned.columns}, inplace=True)

    self.state.df_aligned = df_aligned
    self.state.global_lag = lag

    print(f"Alignment complete. Lag: {lag} frames")
    print(f"df_aligned shape: {df_aligned.shape}")
    print(f"df_aligned columns: {list(df_aligned.columns)}")
    print("[DEBUG] Current Sex in globalVariable:", global_variable.globalVariable.sex)

    # ======================================================================
    # STEP 2: FIND FORCE BOUNDARIES (TRIMMING)
    # ======================================================================
    print("\n[STEP 2] Finding force boundaries for trimming...")
    try:
        boundary_start, boundary_end = find_force_boundaries(
            df_aligned,
            threshold=FORCE_THRESHOLD,
            padding_frames=BOUNDARY_PADDING
        )
        self.state.boundary_start = boundary_start
        self.state.boundary_end = boundary_end
        print(f"Processing subset: frames {boundary_start} to {boundary_end}")

    except Exception as e:
        print(f"[ERROR] Failed to find force boundaries: {e}")
        print("[INFO] Using full frame range as fallback")
        boundary_start = int(df_aligned['FrameNumber'].min())
        boundary_end = int(df_aligned['FrameNumber'].max())

    # ======================================================================
    # STEP 3: RUN COM CALCULATION ON TRIMMED SUBSET
    # ======================================================================
    com_csv_path = None
    if view != "Top View":
        print("\n[STEP 3] Running COM calculation on trimmed subset...")
        com_csv_path = os.path.join(parent_path, "pose_landmarks.csv")

        try:
            print("[DEBUG] Current Sex in globalVariable:", global_variable.globalVariable.sex)
            sex = global_variable.globalVariable.sex if global_variable.globalVariable.sex else 'm'

            if not global_variable.globalVariable.sex:
                print("[WARNING] Sex not set in globalVariable, defaulting to 'male'")

            processor = Processor(video.path)
            print("Processor class:", Processor)
            print("SaveToTxt signature:", Processor.SaveToTxt.__code__.co_varnames)

            processor.SaveToTxt(
                sex=sex,
                filename=com_csv_path,
                confidencelevel=0.85,
                displayCOM=True,
                start_frame=boundary_start,
                end_frame=boundary_end,
                max_workers=MAX_COM_WORKERS
            )

            if com_csv_path and os.path.exists(com_csv_path):
                from Util.COM_helper import COM_helper
                self.COM_helper = COM_helper(com_csv_path)
                print(f"[INFO] COM_helper updated with: {com_csv_path}")

        except Exception as e:
            print(f"[ERROR] COM calculation failed: {e}")
            import traceback
            traceback.print_exc()
            com_csv_path = None

    # ======================================================================
    # STEP 4: RUN VECTOR OVERLAY WITH COM ON TRIMMED SUBSET
    # ======================================================================
    print("\n[STEP 4] Running vector overlay with COM visualization...")

    df_trimmed = get_trimmed_subset(self.state.df_aligned, self.state.boundary_start, self.state.boundary_end)
    temp_video = "vector_overlay_temp.mp4"

    try:
        video.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

        v = VectorOverlay(data=df_trimmed, video=video.cam, view=view)
        v.check_corner(view)

        column_rename2 = {
            'Fx1': 'FP1_Fx', 'Fy1': 'FP1_Fy', 'Fz1': 'FP1_Fz', 'Ft1': 'FP1_|F|',
            'Ax1': 'FP1_Ax', 'Ay1': 'FP1_Ay',
            'Fx2': 'FP2_Fx', 'Fy2': 'FP2_Fy', 'Fz2': 'FP2_Fz', 'Ft2': 'FP2_|F|',
            'Ax2': 'FP2_Ax', 'Ay2': 'FP2_Ay',
            'Fx3': 'FP3_Fx', 'Fy3': 'FP3_Fy', 'Fz3': 'FP3_Fz', 'Ft3': 'FP3_|F|',
            'Ax3': 'FP3_Ax', 'Ay3': 'FP3_Ay',
            'abs time (s)': 'Time(s)'
        }
        df_trimmed.rename(columns={k: v for k, v in column_rename2.items()
                                    if k in df_trimmed.columns}, inplace=True)
        self.state.df_trimmed = df_trimmed.reset_index(drop=True)

        print("[COLUMNS BEFORE] LongVectorOverlay:", list(df_trimmed.columns))

        frames = []
        if view == "Long View":
            frames = v.LongVectorOverlay(
                df_aligned=df_trimmed,
                outputName=temp_video,
                lag=lag,
                com_csv_path=com_csv_path,
                show_landmarks=SHOW_LANDMARKS,
                boundary_start=self.state.boundary_start,
                boundary_end=self.state.boundary_end
            )
        elif view == "Side1 View":
            frames = v.SideVectorOverlay(
                df_aligned=df_trimmed,
                outputName=temp_video,
                lag=lag,
                com_csv_path=com_csv_path,
                show_landmarks=SHOW_LANDMARKS,
                boundary_start=self.state.boundary_start,
                boundary_end=self.state.boundary_end,
                is_side1=True
            )
        elif view == "Side2 View":
            frames = v.SideVectorOverlay(
                df_aligned=df_trimmed,
                outputName=temp_video,
                lag=lag,
                com_csv_path=com_csv_path,
                show_landmarks=SHOW_LANDMARKS,
                boundary_start=self.state.boundary_start,
                boundary_end=self.state.boundary_end,
                is_side1=False
            )
        elif view == "Top View":
            frames = v.TopVectorOverlay(
                df_aligned=df_trimmed,
                outputName=temp_video,
                lag=lag,
                com_csv_path=None,
                show_landmarks=SHOW_LANDMARKS,
                boundary_start=self.state.boundary_start,
                boundary_end=self.state.boundary_end
            )

        print(f"[INFO] Vector overlay complete: {temp_video}")

        video.vector_cam = cv2.VideoCapture(temp_video)
        video.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

        video.vector_cam.set(cv2.CAP_PROP_POS_FRAMES, self.state.loc)
        self.canvasManager.photo_image3 = self.frameConverter.cvToPillow(
            camera=video.vector_cam
        )
        self.canvasManager.canvas3.create_image(
            200, 150,
            image=self.canvasManager.photo_image3,
            anchor="center"
        )

        self.state.vector_overlay_enabled = True
        print("[SUCCESS] All processing complete!")
        return frames

    except Exception as e:
        print(f"[ERROR] Vector overlay failed: {e}")
        import traceback
        traceback.print_exc()