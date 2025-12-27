# Import libraries
import cv2
import cv2 as cv
from matplotlib import lines
import pandas as pd
from vector_overlay.select_corners import select_points
import numpy as np
import os
from GUI.models.corner_state import CornerState

def rect_to_trapezoid(x, y, rect_width, rect_height, trapezoid_coords, short=False):
    """
    Maps points from a rectangle to a trapezoid, simulating parallax distortion.
    """
    # Ensure input coordinates are within the rectangle
    x = np.clip(x, 0, rect_width)
    y = np.clip(y, 0, rect_height)

    # Extract trapezoid coordinates
    (tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y) = trapezoid_coords

    # Calculate the left and right edge positions for the current y
    left_x = bl_x + (tl_x - bl_x) * y
    right_x = br_x + (tr_x - br_x) * y

    # Calculate the width of the trapezoid at the current y
    trapezoid_width = right_x - left_x

    # Map x-coordinate
    new_x = left_x + x * trapezoid_width

    # Calculate the top and bottom y positions of the trapezoid
    top_y = (tl_y + tr_y) / 2
    bottom_y = (bl_y + br_y) / 2

    # Map y-coordinate
    if short:
        new_y = bottom_y + y * (top_y - bottom_y)
    else:
        new_y = top_y + y * (bottom_y - top_y)

    return (int(new_x), int(new_y))

class VectorOverlay:

    def __init__(self, data, video, force_fps=None):
        """
        Initialize VectorOverlay with improved synchronization options.

        Args:
            data: pandas DataFrame with force data
            video: cv2.VideoCapture object
            force_fps: Force data sampling rate (if None, will be calculated)
        """
        # Rename columns for consistency
        # rename_dict = {
        #     "Fx": "Fx1", "Fy": "Fy1", "Fz": "Fz1", "|Ft|": "Ft1", "Ax": "Ax1", "Ay": "Ay1",
        #     "Fx.1": "Fx2", "Fy.1": "Fy2", "Fz.1": "Fz2", "|Ft|.1": "Ft2", "Ax.1": "Ax2", "Ay.1": "Ay2",
        #     "Fx.2": "Fx3", "Fy.2": "Fy3", "Fz.2": "Fz3", "|Ft|.2": "Ft3", "Ax.2": "Ax3", "Ay.2": "Ay3"
        # }

        rename_dict = {
            'Time(s)': 'abs time (s)',
            'FP1_Fx': 'Fx1',
            'FP1_Fy': 'Fy1',
            'FP1_Fz': 'Fz1',
            'FP1_|F|': 'Ft1',
            'FP1_Ax': 'Ax1',
            'FP1_Ay': 'Ay1',

            'FP2_Fx': 'Fx2',
            'FP2_Fy': 'Fy2',
            'FP2_Fz': 'Fz2',
            'FP2_|F|': 'Ft2',
            'FP2_Ax': 'Ax2',
            'FP2_Ay': 'Ay2',

            'FP3_Fx': 'Fx3',
            'FP3_Fy': 'Fy3',
            'FP3_Fz': 'Fz3',
            'FP3_|F|': 'Ft3',
            'FP3_Ax': 'Ax3',
            'FP3_Ay': 'Ay3',
        }

        for key in rename_dict:
            if key in data.columns:
                data.rename(columns={key: rename_dict[key]}, inplace=True)

        self.data = data
        self.video = video
        self.force_fps = force_fps

        print("========== VectorOverlay INIT ==========")
        print(f"Input force DataFrame shape: {data.shape}")
        print(f"Input force DataFrame columns: {list(data.columns)}")
        print(f"Provided force_fps argument: {force_fps}")
        print("========================================")

        # Video properties
        self.frame_width = None
        self.frame_height = None
        self.fps = None
        self.frame_count = None

        # Force data arrays
        self.fx1 = ()
        self.fy1 = ()
        self.fz1 = ()
        self.px1 = ()
        self.py1 = ()
        self.fx2 = ()
        self.fy2 = ()
        self.fz2 = ()
        self.px2 = ()
        self.py2 = ()

        # self.corners = [] OLD
        self.corner_state = CornerState()

        # Initialize
        self.setFrameData()
        self.readData()

    def check_corner(self, view):
        corners = select_points(self, cap=self.video, view=view)
        self.corner_state.set_corners(corners)


    def setFrameData(self):
        cap = self.video

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        self.frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv.CAP_PROP_FPS))
        self.frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        print(f"Frame width: {self.frame_width}, Frame height: {self.frame_height}")
        print(f"Video FPS: {self.fps}, Frame count: {self.frame_count}")

    def normalizeForces(self, x1, x2, y1, y2):
        """Normalize forces for better visualization"""
        max_force = max(
            max(abs(value) for value in x1 if value != 0),
            max(abs(value) for value in x2 if value != 0),
            max(abs(value) for value in y1 if value != 0),
            max(abs(value) for value in y2 if value != 0)
        )

        if max_force > 0:
            scale_factor = min(self.frame_height, self.frame_width) * 0.8 / max_force
        else:
            scale_factor = 1.0

        self.fx1 = tuple(f * scale_factor for f in self.fx1)
        self.fy1 = tuple(f * scale_factor for f in self.fy1)
        self.fz1 = tuple(f * scale_factor for f in self.fz1)
        self.fx2 = tuple(f * scale_factor for f in self.fx2)
        self.fy2 = tuple(f * scale_factor for f in self.fy2)
        self.fz2 = tuple(f * scale_factor for f in self.fz2)

    def readData(self):
        force_samples = len(self.data)
        video_frames = self.frame_count
        video_duration = video_frames / self.fps if self.fps else 0.0

        print("\n========== readData START ==========")
        print(f"Force rows: {force_samples}")
        print(f"Video frames: {video_frames}")
        print(f"Video FPS: {self.fps}")
        print(f"Video duration (s): {video_duration:.3f}")

        # --- determine force sampling frequency from Time(s) or abs time (s) ---
        time_col = None
        if "Time(s)" in self.data.columns:
            time_col = "Time(s)"
        elif "abs time (s)" in self.data.columns:
            time_col = "abs time (s)"

        if time_col is not None:
            print(f"Using time column for force data: '{time_col}'")
        else:
            print("WARNING: No time column found (Time(s) or abs time (s)).")

        if time_col is not None and self.force_fps is None:
            t = self.data[time_col].to_numpy().astype(float)
            if len(t) > 1:
                dt = np.diff(t)
                dt_pos = dt[dt > 0]
                print(f"Force time: t[0]={t[0]:.6f}, t[-1]={t[-1]:.6f}, duration={t[-1]-t[0]:.6f}s")
                if len(dt) > 0:
                    print(f"dt stats: min={dt.min():.6e}, max={dt.max():.6e}, median={np.median(dt):.6e}")
                if len(dt_pos) > 0:
                    self.force_fps = 1.0 / np.median(dt_pos)
                    print(f"Estimated force FPS from {time_col}: {self.force_fps:.2f} Hz")
                else:
                    print("WARNING: No positive dt in time column, cannot estimate force_fps from time.")
                    self.force_fps = self.fps * 10
                    print(f"Defaulting force_fps to {self.force_fps:.2f} Hz (10× video fps)")
            else:
                print("WARNING: Not enough time samples to estimate force_fps; defaulting to 10× video fps.")
                self.force_fps = self.fps * 10
        elif self.force_fps is not None:
            print(f"Using provided force_fps: {self.force_fps:.2f} Hz")
        else:
            # no time column and no force_fps → fall back to heuristic
            if video_duration > 0:
                self.force_fps = force_samples / video_duration
            else:
                self.force_fps = self.fps
            print(f"No time column and no force_fps provided; "
                  f"approximated force_fps from length/duration: {self.force_fps:.2f} Hz")

        # --- TRUE samples per frame based on frequency ratio ---
        samples_per_frame = self.force_fps / self.fps if self.fps > 0 else 1.0
        print(f"Computed samples_per_frame = force_fps / video_fps = "
              f"{self.force_fps:.3f} / {self.fps:.3f} = {samples_per_frame:.3f}")

        if samples_per_frame < 0.1:
            print("WARNING: samples_per_frame < 0.1 → forces will change VERY slowly.")
        if samples_per_frame > 50:
            print("WARNING: samples_per_frame > 50 → forces will change VERY quickly.")

        # Debug: what force index do we hit at several key frames?
        debug_frames = [0,
                        video_frames // 4,
                        video_frames // 2,
                        3 * video_frames // 4,
                        video_frames - 1]
        debug_frames = [f for f in debug_frames if 0 <= f < video_frames]

        if debug_frames:
            print("\nMapping some video frames → force indices:")
            if time_col is not None:
                t = self.data[time_col].to_numpy().astype(float)
            else:
                t = None

            for f_idx in debug_frames:
                data_idx = int(f_idx * samples_per_frame)
                if 0 <= data_idx < force_samples:
                    msg = f"  frame {f_idx:6d} → force idx {data_idx:6d}"
                    if t is not None:
                        video_t = f_idx / self.fps
                        force_t = t[data_idx]
                        msg += f" (video_t={video_t:.4f}s, force_t={force_t:.4f}s, Δ={force_t-video_t:.4f}s)"
                    print(msg)
                else:
                    print(f"  frame {f_idx:6d} → force idx {data_idx:6d} (OUT OF RANGE)")
        print("=====================================\n")

        # ----------------- build per-frame arrays -----------------
        fx1, fy1, fz1, px1, py1 = [], [], [], [], []
        fx2, fy2, fz2, px2, py2 = [], [], [], [], []

        for frame_idx in range(video_frames):
            data_idx = int(frame_idx * samples_per_frame)

            # Ensure index is within bounds
            if 0 <= data_idx < len(self.data):
                row = self.data.iloc[data_idx]

                data_x1 = row.get("Fx1", 0.0) if not pd.isna(row.get("Fx1")) else 0.0
                data_y1 = row.get("Fy1", 0.0) if not pd.isna(row.get("Fy1")) else 0.0
                data_z1 = row.get("Fz1", 0.0) if not pd.isna(row.get("Fz1")) else 0.0

                ax1 = row.get("Ax1", 0.0) if not pd.isna(row.get("Ax1")) else 0.0
                ay1 = row.get("Ay1", 0.0) if not pd.isna(row.get("Ay1")) else 0.0
                pressure_x1 = np.clip((ax1 + 0.3) / 0.6, 0, 1)
                pressure_y1 = np.clip((ay1 + 0.45) / 0.9, 0, 1)

                data_x2 = row.get("Fx2", 0.0) if not pd.isna(row.get("Fx2")) else 0.0
                data_y2 = row.get("Fy2", 0.0) if not pd.isna(row.get("Fy2")) else 0.0
                data_z2 = row.get("Fz2", 0.0) if not pd.isna(row.get("Fz2")) else 0.0

                ax2 = row.get("Ax2", 0.0) if not pd.isna(row.get("Ax2")) else 0.0
                ay2 = row.get("Ay2", 0.0) if not pd.isna(row.get("Ay2")) else 0.0
                pressure_x2 = np.clip((ax2 + 0.3) / 0.6, 0, 1)
                pressure_y2 = np.clip((ay2 + 0.45) / 0.9, 0, 1)
            else:
                data_x1 = data_y1 = data_z1 = 0.0
                pressure_x1 = pressure_y1 = 0.5
                data_x2 = data_y2 = data_z2 = 0.0
                pressure_x2 = pressure_y2 = 0.5

            fx1.append(data_x1)
            fy1.append(data_y1)
            fz1.append(data_z1)
            px1.append(pressure_x1)
            py1.append(pressure_y1)

            fx2.append(data_x2)
            fy2.append(data_y2)
            fz2.append(data_z2)
            px2.append(pressure_x2)
            py2.append(pressure_y2)

        self.fx1 = tuple(fx1)
        self.fy1 = tuple(fy1)
        self.fz1 = tuple(fz1)
        self.px1 = tuple(px1)
        self.py1 = tuple(py1)

        self.fx2 = tuple(fx2)
        self.fy2 = tuple(fy2)
        self.fz2 = tuple(fz2)
        self.px2 = tuple(px2)
        self.py2 = tuple(py2)

        print("readData finished: built force arrays with length:", len(self.fx1))
        print("=====================================\n")

    def drawArrows(self, frame, xf1, xf2, yf1, yf2, px1, px2, py1, py2, short=False):
        """Draw force arrows on frame"""
        if short:
            point_pair1 = rect_to_trapezoid(px1, py1, 1, 1,
                                           [self.corner_state.get_all_corners()[0], self.corner_state.get_all_corners()[1], self.corner_state.get_all_corners()[2], self.corner_state.get_all_corners()[3]], short=True)
            point_pair2 = rect_to_trapezoid(px2, py2, 1, 1,
                                           [self.corner_state.get_all_corners()[4], self.corner_state.get_all_corners()[5], self.corner_state.get_all_corners()[6], self.corner_state.get_all_corners()[7]], short=True)
        else:
            point_pair1 = rect_to_trapezoid(px1, py1, 1, 1,
                                           [self.corner_state.get_all_corners()[0], self.corner_state.get_all_corners()[1], self.corner_state.get_all_corners()[2], self.corner_state.get_all_corners()[3]])
            point_pair2 = rect_to_trapezoid(px2, py2, 1, 1,
                                           [self.corner_state.get_all_corners()[4], self.corner_state.get_all_corners()[5], self.corner_state.get_all_corners()[6], self.corner_state.get_all_corners()[7]])

        end_point_1 = (int(point_pair1[0] + xf1), int(point_pair1[1] - yf1))
        end_point_2 = (int(point_pair2[0] + xf2), int(point_pair2[1] - yf2))

        # Draw arrows with different colors for each plate
        cv.arrowedLine(frame, point_pair1, end_point_1, (0, 255, 0), 4)  # Green for plate 1
        cv.arrowedLine(frame, point_pair2, end_point_2, (255, 0, 0), 4)  # Blue for plate 2

    def scale_factor(self, x1, x2, y1, y2):
        max_force = max(
            max(abs(value) for value in x1),
            max(abs(value) for value in x2),
            max(abs(value) for value in y1),
            max(abs(value) for value in y2)
        )
        
        return min(self.frame_height, self.frame_width) * 0.3 / max_force
    


    def LongVectorOverlay_WithCOM(self, df_trimmed, outputName, start_frame, end_frame,
                                    apply_com=False, com_helper=None):
        """
        Process original video within trim boundaries, applying vectors and COM in one pass.

        Args:
            df_trimmed: Filtered and renumbered df_aligned (FrameNumber 0-based for output)
            outputName: Output video filename
            start_frame: First frame to process from original video
            end_frame: Last frame to process from original video
            apply_com: Whether to apply COM overlay
            com_helper: COM_helper instance for drawing COM points
        """
        print("\n========== LongVectorOverlay_WithCOM START ==========")
        print(f"Processing original video frames {start_frame} to {end_frame}")
        print(f"Output will have {len(df_trimmed)} frames (0-{len(df_trimmed)-1})")
        print(f"Apply COM: {apply_com}")

        # Compute scale factor
        F1_Fy = df_trimmed['FP1_Fy'].astype(float).fillna(0.0).to_numpy()
        F1_Fz = df_trimmed['FP1_Fz'].astype(float).fillna(0.0).to_numpy()
        F2_Fy = df_trimmed['FP2_Fy'].astype(float).fillna(0.0).to_numpy()
        F2_Fz = df_trimmed['FP2_Fz'].astype(float).fillna(0.0).to_numpy()

        scale_factor = self.scale_factor(F1_Fy, F2_Fy, F1_Fz, F2_Fz)
        print(f"Scale factor: {scale_factor:.3f}")

        # Set video to start frame
        self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Create video writer
        out = cv2.VideoWriter(
            outputName,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.frame_width, self.frame_height)
        )

        processed = 0
        output_frame_idx = 0  # Frame number in output video (0-based)

        # Process each row in df_trimmed
        for idx, row in df_trimmed.iterrows():
            original_frame_idx = int(row["OriginalFrameNumber"])

            # Read frame from original video at exact position
            self.video.set(cv2.CAP_PROP_POS_FRAMES, original_frame_idx)
            ret, frame = self.video.read()
            if not ret:
                print(f"[WARN] Could not read frame {original_frame_idx}, stopping.")
                break

            # Extract forces (same as before)
            raw_F1_Fy = float(row.get("FP1_Fy", 0.0) or 0.0)
            raw_F1_Fz = float(row.get("FP1_Fz", 0.0) or 0.0)
            raw_F2_Fy = float(row.get("FP2_Fy", 0.0) or 0.0)
            raw_F2_Fz = float(row.get("FP2_Fz", 0.0) or 0.0)

            fx1 = -raw_F1_Fy * scale_factor
            fy1 =  raw_F1_Fz * scale_factor
            fx2 = -raw_F2_Fy * scale_factor
            fy2 =  raw_F2_Fz * scale_factor

            # Extract pressure coordinates
            ax1 = float(row.get("FP1_Ax", 0.0) or 0.0)
            ay1 = float(row.get("FP1_Ay", 0.0) or 0.0)
            ax2 = float(row.get("FP2_Ax", 0.0) or 0.0)
            ay2 = float(row.get("FP2_Ay", 0.0) or 0.0)

            pressure_x1 = np.clip((ax1 + 0.3) / 0.6, 0, 1)
            pressure_y1 = np.clip((ay1 + 0.45) / 0.9, 0, 1)
            pressure_x2 = np.clip((ax2 + 0.3) / 0.6, 0, 1)
            pressure_y2 = np.clip((ay2 + 0.45) / 0.9, 0, 1)

            # Swap for long view
            px1 = pressure_y1
            py1 = pressure_x1
            px2 = pressure_y2
            py2 = pressure_x2

            # Draw force arrows
            self.drawArrows(frame, fx1, fx2, fy1, fy2, px1, px2, py1, py2)

            # Apply COM overlay if enabled
            if apply_com and com_helper is not None:
                try:
                    # Use output_frame_idx (0-based) for COM data
                    frame = com_helper.drawFigure(frame, output_frame_idx)
                except Exception as e:
                    if processed < 5:  # Only print first few errors
                        print(f"[WARN] Could not apply COM at frame {output_frame_idx}: {e}")

            out.write(frame)
            processed += 1
            output_frame_idx += 1

            if processed % 100 == 0:
                print(f"  Processed {processed}/{len(df_trimmed)} frames...")

        out.release()
        print(f"Finished; Total frames written: {processed}")
        print("========== LongVectorOverlay_WithCOM END ==========\n")


    def ShortVectorOverlay_WithCOM(self, df_trimmed, outputName, start_frame, end_frame,
                                     apply_com=False, com_helper=None):
        """Short view with COM overlay in single pass."""
        print("\n========== ShortVectorOverlay_WithCOM START ==========")
        print(f"Processing original video frames {start_frame} to {end_frame}")

        # Determine which plate is in front
        all_corners = self.corner_state.get_all_corners()
        plate2_in_front = all_corners[0][1] < all_corners[4][1]
        print(f"Force plate {'2' if plate2_in_front else '1'} is in front")

        # Compute scale factor (only from plate 2)
        scale_factor = self.scale_factor(
            [0.0],
            df_trimmed["FP2_Fx"].astype(float).fillna(0.0).to_numpy(),
            [0.0],
            df_trimmed["FP2_Fz"].astype(float).fillna(0.0).to_numpy(),
        )

        out = cv2.VideoWriter(
            outputName,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.frame_width, self.frame_height)
        )

        processed = 0
        output_frame_idx = 0

        for _, row in df_trimmed.iterrows():
            original_frame_idx = int(row["OriginalFrameNumber"])

            self.video.set(cv2.CAP_PROP_POS_FRAMES, original_frame_idx)
            ret, frame = self.video.read()
            if not ret:
                break

            # Extract forces
            Fx1 = float(row.get("FP1_Fx", 0.0) or 0.0)
            Fz1 = float(row.get("FP1_Fz", 0.0) or 0.0)
            Fx2 = float(row.get("FP2_Fx", 0.0) or 0.0)
            Fz2 = float(row.get("FP2_Fz", 0.0) or 0.0)

            # Extract pressures
            Ax1 = float(row.get("FP1_Ax", 0.0) or 0.0)
            Ay1 = float(row.get("FP1_Ay", 0.0) or 0.0)
            Ax2 = float(row.get("FP2_Ax", 0.0) or 0.0)
            Ay2 = float(row.get("FP2_Ay", 0.0) or 0.0)

            px1 = np.clip((Ax1 + 0.3) / 0.6, 0, 1)
            py1 = np.clip((Ay1 + 0.45) / 0.9, 0, 1)
            px2 = np.clip((Ax2 + 0.3) / 0.6, 0, 1)
            py2 = np.clip((Ay2 + 0.45) / 0.9, 0, 1)

            # Apply short view logic
            if plate2_in_front:
                fx1 = -Fx1 * scale_factor
                fx2 = -Fx2 * scale_factor
                fy1 =  Fz1 * scale_factor
                fy2 =  Fz2 * scale_factor
                py1 = 1 - py1
                py2 = 1 - py2
            else:
                fx1 =  Fx1 * scale_factor
                fx2 =  Fx2 * scale_factor
                fy1 =  Fz1 * scale_factor
                fy2 =  Fz2 * scale_factor
                px1 = 1 - px1
                px2 = 1 - px2
                py1 = 1 - py1
                py2 = 1 - py2

            self.drawArrows(frame, fx1, fx2, fy1, fy2, px1, px2, py1, py2, short=True)

            # Apply COM
            if apply_com and com_helper is not None:
                try:
                    frame = com_helper.drawFigure(frame, output_frame_idx)
                except Exception as e:
                    if processed < 5:
                        print(f"[WARN] Could not apply COM: {e}")

            out.write(frame)
            processed += 1
            output_frame_idx += 1

        out.release()
        print(f"Finished; Total frames: {processed}")
        print("========== ShortVectorOverlay_WithCOM END ==========\n")


    def TopVectorOverlay_WithCOM(self, df_trimmed, outputName, start_frame, end_frame,
                                   apply_com=False, com_helper=None):
        """Top view with COM overlay in single pass."""
        print("\n========== TopVectorOverlay_WithCOM START ==========")
        print(f"Processing original video frames {start_frame} to {end_frame}")

        # Compute scale factor
        F1_Fx = df_trimmed["FP1_Fx"].astype(float).fillna(0.0).to_numpy()
        F1_Fy = df_trimmed["FP1_Fy"].astype(float).fillna(0.0).to_numpy()
        F2_Fx = df_trimmed["FP2_Fx"].astype(float).fillna(0.0).to_numpy()
        F2_Fy = df_trimmed["FP2_Fy"].astype(float).fillna(0.0).to_numpy()

        scale_factor = self.scale_factor(F1_Fy, F2_Fy, F1_Fx, F2_Fx)

        out = cv2.VideoWriter(
            outputName,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.frame_width, self.frame_height),
        )

        processed = 0
        output_frame_idx = 0

        for _, row in df_trimmed.iterrows():
            original_frame_idx = int(row["OriginalFrameNumber"])

            self.video.set(cv2.CAP_PROP_POS_FRAMES, original_frame_idx)
            ret, frame = self.video.read()
            if not ret:
                break

            # Extract forces
            raw_F1_Fx = float(row.get("FP1_Fx", 0.0) or 0.0)
            raw_F1_Fy = float(row.get("FP1_Fy", 0.0) or 0.0)
            raw_F2_Fx = float(row.get("FP2_Fx", 0.0) or 0.0)
            raw_F2_Fy = float(row.get("FP2_Fy", 0.0) or 0.0)

            fx1 = -raw_F1_Fy * scale_factor
            fx2 = -raw_F2_Fy * scale_factor
            fy1 = -raw_F1_Fx * scale_factor
            fy2 = -raw_F2_Fx * scale_factor

            # Extract pressures
            ax1 = float(row.get("FP1_Ax", 0.0) or 0.0)
            ay1 = float(row.get("FP1_Ay", 0.0) or 0.0)
            ax2 = float(row.get("FP2_Ax", 0.0) or 0.0)
            ay2 = float(row.get("FP2_Ay", 0.0) or 0.0)

            pressure_x1 = np.clip((ax1 + 0.3) / 0.6, 0, 1)
            pressure_y1 = np.clip((ay1 + 0.45) / 0.9, 0, 1)
            pressure_x2 = np.clip((ax2 + 0.3) / 0.6, 0, 1)
            pressure_y2 = np.clip((ay2 + 0.45) / 0.9, 0, 1)

            px1 = pressure_y1
            py1 = 1.0 - pressure_x1
            px2 = pressure_y2
            py2 = 1.0 - pressure_x2

            self.drawArrows(frame, fx1, fx2, fy1, fy2, px1, px2, py1, py2)

            # Apply COM
            if apply_com and com_helper is not None:
                try:
                    frame = com_helper.drawFigure(frame, output_frame_idx)
                except Exception as e:
                    if processed < 5:
                        print(f"[WARN] Could not apply COM: {e}")

            out.write(frame)
            processed += 1
            output_frame_idx += 1

        out.release()
        print(f"Finished; Total frames: {processed}")
        print("========== TopVectorOverlay_WithCOM END ==========\n")

# Example usage with synchronization parameters
if __name__ == "__main__":
    # Example with time offset to reduce delay
    # df = pd.read_excel("your_data.xlsx", skiprows=19)
    # cap = cv2.VideoCapture("your_video.mp4")
    #
    # # Try different time offsets to find the best sync
    # # Positive offset = data leads video, Negative = data lags video
    # v = VectorOverlay(df, cap, time_offset=-0.1, force_fps=1000)  # Adjust as needed
    # v.LongVectorOverlay("output.mp4", show_preview=True)
    print("Vector overlay module loaded. Use with appropriate data and video files.")