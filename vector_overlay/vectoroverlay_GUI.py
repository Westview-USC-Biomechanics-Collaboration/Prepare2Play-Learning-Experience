# Import libraries
import cv2
import cv2 as cv
from matplotlib import lines
import pandas as pd
from vector_overlay.select_corners import select_points
import numpy as np
import os
from Util.COM_helper import COM_helper


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

    def __init__(self, data, video, force_fps=None, view=None):
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
        self.view = view

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

        self.corners = []
        self.com_data = None # NEW
        self.com_helper = None

        # Initialize
        self.setFrameData()
        self.readData()

    # def check_corner(self, view):
    #     self.corners = select_points(self, cap=self.video, view=view)

    def check_corner(self, view):
        from vector_overlay.select_corners import select_points_with_manual_adjustment
        self.corners = select_points_with_manual_adjustment(self, cap=self.video, view=view)

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
    
    # NEW METHODS
    def load_com_helper(self, com_csv_path):
        if not com_csv_path:
            print("[VectorOverlay] No COM CSV provided")
            self.com_helper = None
            return

        self.com_helper = COM_helper(com_csv_path)

    # def load_com_data(self, com_csv_path):
    #     """
    #     Load COM data from CSV file.
        
    #     Args:
    #         com_csv_path: Path to CSV file with COM data
    #     """
    #     if com_csv_path is None or not com_csv_path:
    #         print("[VectorOverlay] No COM data provided")
    #         self.com_data = None
    #         return
        
    #     try:
    #         self.com_data = pd.read_csv(com_csv_path)
    #         print(f"[VectorOverlay] Loaded COM data: {len(self.com_data)} rows")
            
    #         # Verify required columns exist
    #         required_cols = ['frame_index', 'COM_x', 'COM_y']
    #         missing = [col for col in required_cols if col not in self.com_data.columns]
            
    #         if missing:
    #             print(f"[VectorOverlay WARNING] Missing columns: {missing}")
    #             self.com_data = None
    #             return
            
    #         # Print frame range for debugging
    #         min_frame = self.com_data['frame_index'].min()
    #         max_frame = self.com_data['frame_index'].max()
    #         print(f"[VectorOverlay] COM frame range: {min_frame} to {max_frame}")
            
    #         # Create frame index lookup for fast access
    #         self.com_data.set_index('frame_index', inplace=True)
            
        # except Exception as e:
        #     print(f"[VectorOverlay ERROR] Failed to load COM data: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     self.com_data = None
    
    def draw_com_on_frame(self, frame, frame_number, debug=True):
        """
        Draw COM point on the frame if COM data is available.
        
        Args:
            frame: The frame to draw on (will be modified in place)
            frame_number: The current frame number
            debug: If True, print debug info
        
        Returns:
            frame: Frame with COM point drawn
        """
        if self.com_data is None:
            if debug and frame_number % 30 == 0:
                print(f"[VectorOverlay] Frame {frame_number}: No COM data loaded")
            return frame
        
        try:
            # Try to find COM data for this frame
            if frame_number not in self.com_data.index:
                if debug and frame_number % 30 == 0:
                    print(f"[VectorOverlay] Frame {frame_number}: Not in COM data index")
                return frame

            com_row = self.com_data.loc[frame_number]
            com_x = float(com_row['x'])
            com_y = float(com_row['y'])
            
            # Skip if no data (0, 0)
            if com_x == 0 and com_y == 0:
                return frame
            
            frame = frame.copy()
            height, width = frame.shape[:2]
            
            # Convert to pixel coordinates if necessary
            if self._coords_are_normalized:
                pixel_x = int(com_x * width)
                pixel_y = int(com_y * height)
            else:
                # Already in pixel coordinates
                pixel_x = int(com_x)
                pixel_y = int(com_y)
            
            # Clamp to frame boundaries
            pixel_x = max(0, min(pixel_x, width - 1))
            pixel_y = max(0, min(pixel_y, height - 1))
            
            # Draw COM marker (red circle)
            if pixel_x > 0 and pixel_y > 0:
                cv2.circle(frame, (pixel_x, pixel_y), 12, (0, 0, 255), -1)
                # Only print occasionally to avoid spam
                if row % 30 == 0:
                    print(f"[COM_helper] Frame {row}: Drew COM at pixel ({pixel_x}, {pixel_y}) from raw ({com_x:.2f}, {com_y:.2f})")
            
            # # Get COM coordinates
            # com_row = self.com_data.loc[frame_number]
            # com_x = float(com_row['COM_x'])
            # com_y = float(com_row['COM_y'])
            
            # # CRITICAL: COM was calculated on 0.3x scaled frames
            # # Scale back to full resolution
            # com_x_full = int(com_x / 0.3)
            # com_y_full = int(com_y / 0.3)
            
            # # Verify point is within frame bounds
            # height, width = frame.shape[:2]
            # if 0 <= com_x_full < width and 0 <= com_y_full < height:
            #     # Draw red circle for COM
            #     cv2.circle(frame, (com_x_full, com_y_full), 12, (0, 0, 255), -1)
                
            #     if debug:
            #         print(f"[VectorOverlay] Frame {frame_number}: Drew COM at ({com_x_full}, {com_y_full})")
            # else:
            #     if debug:
            #         print(f"[VectorOverlay] Frame {frame_number}: COM out of bounds: ({com_x_full}, {com_y_full})")
        
        except Exception as e:
            if debug:
                print(f"[VectorOverlay] Frame {frame_number}: Error drawing COM: {e}")
        
        return frame


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
                                           [self.corners[0], self.corners[1], self.corners[2], self.corners[3]], short=True)
            point_pair2 = rect_to_trapezoid(px2, py2, 1, 1,
                                           [self.corners[4], self.corners[5], self.corners[6], self.corners[7]], short=True)
        else:
            point_pair1 = rect_to_trapezoid(px1, py1, 1, 1,
                                           [self.corners[0], self.corners[1], self.corners[2], self.corners[3]])
            point_pair2 = rect_to_trapezoid(px2, py2, 1, 1,
                                           [self.corners[4], self.corners[5], self.corners[6], self.corners[7]])

        end_point_1 = (int(point_pair1[0] + xf1), int(point_pair1[1] - yf1))
        end_point_2 = (int(point_pair2[0] + xf2), int(point_pair2[1] - yf2))

        # Draw arrows with different colors for each plate
        cv.arrowedLine(frame, point_pair1, end_point_1, (0, 255, 0), 4)  # Green for plate 1 
        cv.arrowedLine(frame, point_pair2, end_point_2, (255, 0, 0), 4)  # Blue for plate 2
        if self.view == "Top View":
            cv.arrowedLine(frame, point_pair1, end_point_1, (255, 0 ,200), 4)  # Purple for plate 1
            cv.arrowedLine(frame, point_pair2, end_point_2, (0, 165, 255), 4)  # Orange for plate 2

    def scale_factor(self, x1, x2, y1, y2):
        max_force = max(
            max(abs(value) for value in x1),
            max(abs(value) for value in x2),
            max(abs(value) for value in y1),
            max(abs(value) for value in y2)
        )
        
        return min(self.frame_height, self.frame_width) * 0.3 / max_force

    def LongVectorOverlay(self, df_aligned, outputName=None, show_preview=True,
                          lag=0, com_csv_path=None, show_landmarks=False,
                          boundary_start=0, boundary_end=None):
        """
        Modified to support:
        - com_csv_path: Path to COM CSV file
        - show_landmarks: Whether to show green skeleton dots
        - boundary_start/end: Frame boundaries to process
        """
        # Load COM data
        # self.load_com_data(com_csv_path)
        self.load_com_helper(com_csv_path)

        if boundary_end is None:
            boundary_end = self.frame_count - 1

        """
        Long view vector overlay using df_aligned for exact frame/force mapping.

        - Uses df_aligned['FrameNumber'] to pick video frames
        - Uses df_aligned force columns to compute arrows
        - Draws arrows with the same logic as your original (fx1 = -Fy, fy1 = Fz, etc.)
        - Ignores the old lag/fps-based alignment (df_aligned already includes it)
        """
        print("\n========== LongVectorOverlay (df_aligned) START ==========")
        print(f"df_aligned rows: {len(df_aligned)}")
        print(f"Video frames: {self.frame_count}, fps: {self.fps}")

        if self.frame_width is None or self.frame_height is None:
            print("Error: Frame data not set.")
            return

        if not outputName:
            outputName = "long_view_overlay_output.mp4"
            print(f"No output name provided, using default: {outputName}")

        # -------- 1. Compute a global scale factor from df_aligned forces --------
        # We follow your original pattern: use Fy & Fz from each plate and
        # choose a scale so the biggest vector fits nicely in the frame.

        # Raw forces from df_aligned
        F1_Fy = df_aligned['FP1_Fy'].astype(float).fillna(0.0).to_numpy()
        F1_Fz = df_aligned['FP1_Fz'].astype(float).fillna(0.0).to_numpy()
        F2_Fy = df_aligned['FP2_Fy'].astype(float).fillna(0.0).to_numpy()
        F2_Fz = df_aligned['FP2_Fz'].astype(float).fillna(0.0).to_numpy()

        scale_factor = self.scale_factor(F1_Fy, F2_Fy, F1_Fz, F2_Fz)

        # print(f"max_force from df_aligned = {max_force:.3f}")
        # print(f"Using scale_factor = {scale_factor:.3f}")

        # -------- 2. Prepare video writer --------
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        out = cv.VideoWriter(
            outputName,
            cv.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.frame_width, self.frame_height)
        )

        # Optional: force time column for debug
        if "Time(s)" in df_aligned.columns:
            force_time_array = df_aligned["Time(s)"].astype(float).to_numpy()
            time_col_name = "Time(s)"
        elif "abs time (s)" in df_aligned.columns:
            force_time_array = df_aligned["abs time (s)"].astype(float).to_numpy()
            time_col_name = "abs time (s)"
        else:
            force_time_array = None
            time_col_name = None

        # -------- 3. Main loop: row-by-row using FrameNumber --------
        processed = 0
        com_drawn_count = 0

        for idx, row in df_aligned.iterrows():
            frame_idx = int(row["FrameNumber"])

            # Safety: skip illegal frame indices
            if frame_idx < 0 or frame_idx >= self.frame_count:
                print(f"[WARN] Row {idx}: FrameNumber {frame_idx} out of range, skipping.")
                continue
            
            if frame_idx < boundary_start or frame_idx > boundary_end:
                continue

            # Jump to the *exact* frame for this force sample
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video.read()
            if not ret:
                print(f"[WARN] Could not read frame {frame_idx}, stopping.")
                break

            # ----- Forces (same orientation logic as your original code) -----
            # In your class you do:
            #   fx1 = -self.fy1[force_idx]
            #   fy1 =  self.fz1[force_idx]
            # So here we mimic that: horizontal is -Fy, vertical is Fz.
            raw_F1_Fy = float(row.get("FP1_Fy", 0.0) or 0.0)
            raw_F1_Fz = float(row.get("FP1_Fz", 0.0) or 0.0)
            raw_F2_Fy = float(row.get("FP2_Fy", 0.0) or 0.0)
            raw_F2_Fz = float(row.get("FP2_Fz", 0.0) or 0.0)

            fx1 = -raw_F1_Fy * scale_factor
            fy1 =  raw_F1_Fz * scale_factor
            fx2 = -raw_F2_Fy * scale_factor
            fy2 =  raw_F2_Fz * scale_factor

            # ----- Pressure coordinates → normalized 0–1 as in readData() -----
            # rename_dict in __init__ mapped:
            #   'FP1_Ax' -> 'Ax1', 'FP1_Ay' -> 'Ay1', etc.
            # but since we are using df_aligned *before* that rename,
            # we read directly from 'FP1_Ax', 'FP1_Ay', etc.
             # ----- Pressure coordinates → normalized 0–1 as in readData(), THEN SWAP -----
            ax1 = float(row.get("FP1_Ax", 0.0) or 0.0)
            ay1 = float(row.get("FP1_Ay", 0.0) or 0.0)
            ax2 = float(row.get("FP2_Ax", 0.0) or 0.0)
            ay2 = float(row.get("FP2_Ay", 0.0) or 0.0)

            # Same normalization as readData()
            pressure_x1 = np.clip((ax1 + 0.3) / 0.6, 0, 1)
            pressure_y1 = np.clip((ay1 + 0.45) / 0.9, 0, 1)
            pressure_x2 = np.clip((ax2 + 0.3) / 0.6, 0, 1)
            pressure_y2 = np.clip((ay2 + 0.45) / 0.9, 0, 1)

            # IMPORTANT: mimic original LongVectorOverlay mapping:
            #   px1 = self.py1[force_idx]
            #   py1 = self.px1[force_idx]
            # i.e., swap x/y before rect_to_trapezoid
            px1 = pressure_y1
            py1 = pressure_x1
            px2 = pressure_y2
            py2 = pressure_x2

            # ----- Debug prints (similar style to your previous logs) -----
            if processed < 10 or processed % 30 == 0:
                video_t = frame_idx / self.fps if self.fps else 0.0
                if force_time_array is not None and 0 <= idx < len(force_time_array):
                    force_t = force_time_array[idx]
                    print(
                        f"[DEBUG] row={idx:5d}, frame={frame_idx:5d}, "
                        f"video_t={video_t:.4f}s, {time_col_name}={force_t:.4f}s, "
                        f"Δ={force_t - video_t:.4f}s, "
                        f"F1_Fy={raw_F1_Fy:.2f}, F1_Fz={raw_F1_Fz:.2f}"
                    )
                else:
                    print(
                        f"[DEBUG] row={idx:5d}, frame={frame_idx:5d}, "
                        f"video_t={video_t:.4f}s, "
                        f"F1_Fy={raw_F1_Fy:.2f}, F1_Fz={raw_F1_Fz:.2f}"
                    )

            # ----- Draw arrows exactly the same way as original -----
            self.drawArrows(frame, fx1, fx2, fy1, fy2, px1, px2, py1, py2)

            # ----- Optionally draw landmarks -----
            debug_com = (processed < 10)
            # frame = self.draw_com_on_frame(frame, frame_idx, debug=debug_com)
            if self.com_helper is not None:
                frame = self.com_helper.drawFigure(frame, frame_idx)

            # Track how many frames had COM drawn
            if self.com_data is not None and frame_idx in self.com_data.index:
                com_drawn_count += 1

            # Show preview if desired
            if show_preview:
                preview_frame = cv2.resize(frame, (self.frame_width // 2, self.frame_height // 2))
                cv2.imshow("Long View with COM", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            
            # Write to output video
            out.write(frame)
            processed += 1

        out.release()
        if show_preview:
            cv2.destroyAllWindows()

        print(f"Processed {processed} frames, COM drawn on {com_drawn_count} frames")
        print("=" * 50 + "\n")

    def TopVectorOverlay(self, df_aligned, outputName=None, show_preview=True,
                          lag=0, com_csv_path=None, show_landmarks=False,
                          boundary_start=0, boundary_end=None):
        """
        Top view vector overlay using df_aligned for exact frame/force mapping.

        Uses TOP VIEW conventions:
        - normalize based on Fy and Fx (not Fz)
        - fx = -Fy, fy = -Fx
        - px = pressure_y, py = 1 - pressure_x
        """
        if boundary_end is None:
            boundary_end = self.frame_count - 1

        print("\n========== TopVectorOverlay (df_aligned) START ==========")
        print(f"df_aligned rows: {len(df_aligned)}")
        print(f"Video frames: {self.frame_count}, fps: {self.fps}")

        if self.frame_width is None or self.frame_height is None:
            print("Error: Frame data not set.")
            return

        if not outputName:
            outputName = "top_view_overlay_output.mp4"
            print(f"No output name provided, using default: {outputName}")

        # -------- 1) Compute a scale factor for TOP VIEW (Fx & Fy) --------
        F1_Fx = df_aligned["FP1_Fx"].astype(float).fillna(0.0).to_numpy()
        F1_Fy = df_aligned["FP1_Fy"].astype(float).fillna(0.0).to_numpy()
        F2_Fx = df_aligned["FP2_Fx"].astype(float).fillna(0.0).to_numpy()
        F2_Fy = df_aligned["FP2_Fy"].astype(float).fillna(0.0).to_numpy()

        scale_factor = self.scale_factor(F1_Fy, F2_Fy, F1_Fx, F2_Fx)

        # max_force = max(
        #     float(np.max(np.abs(F1_Fx))) if len(F1_Fx) else 0.0,
        #     float(np.max(np.abs(F1_Fy))) if len(F1_Fy) else 0.0,
        #     float(np.max(np.abs(F2_Fx))) if len(F2_Fx) else 0.0,
        #     float(np.max(np.abs(F2_Fy))) if len(F2_Fy) else 0.0,
        # )

        # if max_force > 0:
        #     scale_factor = min(self.frame_height, self.frame_width) * 0.8 / max_force
        # else:
        #     scale_factor = 1.0

        # print(f"max_force (TOP VIEW, Fx/Fy) = {max_force:.3f}")
        # print(f"Using scale_factor = {scale_factor:.3f}")

        # -------- 2) Video writer --------
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        out = cv.VideoWriter(
            outputName,
            cv.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.frame_width, self.frame_height),
        )

        # Optional debug time column
        if "Time(s)" in df_aligned.columns:
            force_time_array = df_aligned["Time(s)"].astype(float).to_numpy()
            time_col_name = "Time(s)"
        elif "abs time (s)" in df_aligned.columns:
            force_time_array = df_aligned["abs time (s)"].astype(float).to_numpy()
            time_col_name = "abs time (s)"
        else:
            force_time_array = None
            time_col_name = None

        # -------- 3) Main loop: row-by-row using FrameNumber --------
        processed = 0
        for row_i, (idx, row) in enumerate(df_aligned.iterrows()):
            frame_idx = int(row["FrameNumber"])

            if frame_idx < 0 or frame_idx >= self.frame_count:
                print(f"[WARN] Row {idx}: FrameNumber {frame_idx} out of range, skipping.")
                continue
            
            if frame_idx < boundary_start or frame_idx > boundary_end:
                continue

            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video.read()
            if not ret:
                print(f"[WARN] Could not read frame {frame_idx}, stopping.")
                break

            # ----- TOP VIEW force mapping (match your working TopVectorOverlay) -----
            raw_F1_Fx = float(row.get("FP1_Fx", 0.0) or 0.0)
            raw_F1_Fy = float(row.get("FP1_Fy", 0.0) or 0.0)
            raw_F2_Fx = float(row.get("FP2_Fx", 0.0) or 0.0)
            raw_F2_Fy = float(row.get("FP2_Fy", 0.0) or 0.0)

            fx1 = -raw_F1_Fy * scale_factor
            fx2 = -raw_F2_Fy * scale_factor
            fy1 = -raw_F1_Fx * scale_factor
            fy2 = -raw_F2_Fx * scale_factor

            # ----- TOP VIEW pressure mapping (match your working version) -----
            ax1 = float(row.get("FP1_Ax", 0.0) or 0.0)
            ay1 = float(row.get("FP1_Ay", 0.0) or 0.0)
            ax2 = float(row.get("FP2_Ax", 0.0) or 0.0)
            ay2 = float(row.get("FP2_Ay", 0.0) or 0.0)

            pressure_x1 = np.clip((ax1 + 0.3) / 0.6, 0, 1)
            pressure_y1 = np.clip((ay1 + 0.45) / 0.9, 0, 1)
            pressure_x2 = np.clip((ax2 + 0.3) / 0.6, 0, 1)
            pressure_y2 = np.clip((ay2 + 0.45) / 0.9, 0, 1)

            # match:
            #   px = self.py
            #   py = 1 - self.px
            px1 = pressure_y1
            py1 = 1.0 - pressure_x1
            px2 = pressure_y2
            py2 = 1.0 - pressure_x2

            # Debug prints
            if processed < 10 or processed % 30 == 0:
                video_t = frame_idx / self.fps if self.fps else 0.0
                if force_time_array is not None and 0 <= idx < len(force_time_array):
                    force_t = force_time_array[idx]
                    print(
                        f"[DEBUG] row={idx:5d}, frame={frame_idx:5d}, "
                        f"video_t={video_t:.4f}s, {time_col_name}={force_t:.4f}s, Δ={force_t - video_t:.4f}s, "
                        f"F1_Fx={raw_F1_Fx:.2f}, F1_Fy={raw_F1_Fy:.2f}"
                    )
                else:
                    print(
                        f"[DEBUG] row={idx:5d}, frame={frame_idx:5d}, "
                        f"video_t={video_t:.4f}s, "
                        f"F1_Fx={raw_F1_Fx:.2f}, F1_Fy={raw_F1_Fy:.2f}"
                    )

            # Draw arrows using your trapezoid mapping (unchanged)
            self.drawArrows(frame, fx1, fx2, fy1, fy2, px1, px2, py1, py2)

            if show_preview:
                preview_frame = cv2.resize(frame, (self.frame_width // 2, self.frame_height // 2))
                cv2.imshow("Top View (No COM)", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            out.write(frame)
            processed += 1

        out.release()
        if show_preview:
            cv2.destroyAllWindows()

        print(f"Finished processing video; Total Frames written: {processed}")
        print("========== TopVectorOverlay (df_aligned) END ==========\n")
# In vector_overlay/vectoroverlay_GUI.py, add a Side vector overlay method:

    def SideVectorOverlay(self, df_aligned, outputName=None, show_preview=True,
                        lag=0, com_csv_path=None, show_landmarks=False,
                        boundary_start=0, boundary_end=None, is_side1=True):
        """
        Side view vector overlay (handles both Side1 and Side2).
        
        Args:
            is_side1: True if Side1 View (FP1 near), False if Side2 View (FP2 near)
        """
        self.load_com_helper(com_csv_path)

        if boundary_end is None:
            boundary_end = self.frame_count - 1

        print(f"\n========== SideVectorOverlay ({'Side1' if is_side1 else 'Side2'}) ==========")
        print(f"df_aligned rows: {len(df_aligned)}")
        print(f"Video frames: {self.frame_count}, fps: {self.fps}")

        if self.frame_width is None or self.frame_height is None:
            print("Error: Frame data not set.")
            return

        if not outputName:
            outputName = f"side{'1' if is_side1 else '2'}_view_overlay_output.mp4"

        # Force labels for side view (Fx horizontal, Fz vertical)
        label1_1, label1_2 = "FP1_Fx", "FP1_Fz"
        label2_1, label2_2 = "FP2_Fx", "FP2_Fz"
        
        # Extract force data
        F1_Fx = df_aligned["FP1_Fx"].astype(float).fillna(0.0).to_numpy()
        F1_Fz = df_aligned["FP1_Fz"].astype(float).fillna(0.0).to_numpy()
        F2_Fx = df_aligned["FP2_Fx"].astype(float).fillna(0.0).to_numpy()
        F2_Fz = df_aligned["FP2_Fz"].astype(float).fillna(0.0).to_numpy()

        scale_factor = self.scale_factor(F1_Fx, F2_Fx, F1_Fz, F2_Fz)

        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        out = cv.VideoWriter(
            outputName,
            cv.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.frame_width, self.frame_height),
        )

        # Optional time column
        if "Time(s)" in df_aligned.columns:
            force_time_array = df_aligned["Time(s)"].astype(float).to_numpy()
            time_col_name = "Time(s)"
        elif "abs time (s)" in df_aligned.columns:
            force_time_array = df_aligned["abs time (s)"].astype(float).to_numpy()
            time_col_name = "abs time (s)"
        else:
            force_time_array = None
            time_col_name = None

        processed = 0
        for row_i, (idx, row) in enumerate(df_aligned.iterrows()):
            frame_idx = int(row["FrameNumber"])

            if frame_idx < 0 or frame_idx >= self.frame_count:
                print(f"[WARN] Row {idx}: FrameNumber {frame_idx} out of range, skipping.")
                continue
            
            if frame_idx < boundary_start or frame_idx > boundary_end:
                continue

            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video.read()
            if not ret:
                print(f"[WARN] Could not read frame {frame_idx}, stopping.")
                break

            # Side view force mapping: Fx horizontal, Fz vertical
            raw_F1_Fx = float(row.get("FP1_Fx", 0.0) or 0.0)
            raw_F1_Fz = float(row.get("FP1_Fz", 0.0) or 0.0)
            raw_F2_Fx = float(row.get("FP2_Fx", 0.0) or 0.0)
            raw_F2_Fz = float(row.get("FP2_Fz", 0.0) or 0.0)

            fx1 = raw_F1_Fx * scale_factor
            fy1 = raw_F1_Fz * scale_factor
            fx2 = raw_F2_Fx * scale_factor
            fy2 = raw_F2_Fz * scale_factor

            # Pressure mapping for side views
            ax1 = float(row.get("FP1_Ax", 0.0) or 0.0)
            ay1 = float(row.get("FP1_Ay", 0.0) or 0.0)
            ax2 = float(row.get("FP2_Ax", 0.0) or 0.0)
            ay2 = float(row.get("FP2_Ay", 0.0) or 0.0)

            pressure_x1 = np.clip((ax1 + 0.3) / 0.6, 0, 1)
            pressure_y1 = np.clip((ay1 + 0.45) / 0.9, 0, 1)
            pressure_x2 = np.clip((ax2 + 0.3) / 0.6, 0, 1)
            pressure_y2 = np.clip((ay2 + 0.45) / 0.9, 0, 1)

            # For side views, map pressure coordinates
            px1 = pressure_y1
            py1 = 1.0 - pressure_x1
            px2 = pressure_y2
            py2 = 1.0 - pressure_x2

            # Debug prints
            if processed < 10 or processed % 30 == 0:
                video_t = frame_idx / self.fps if self.fps else 0.0
                if force_time_array is not None and 0 <= idx < len(force_time_array):
                    force_t = force_time_array[idx]
                    print(
                        f"[DEBUG] row={idx:5d}, frame={frame_idx:5d}, "
                        f"video_t={video_t:.4f}s, {time_col_name}={force_t:.4f}s, Δ={force_t - video_t:.4f}s, "
                        f"F1_Fx={raw_F1_Fx:.2f}, F1_Fz={raw_F1_Fz:.2f}"
                    )

            # Draw arrows (use short=True for side view trapezoid mapping)
            self.drawArrows(frame, fx1, fx2, fy1, fy2, px1, px2, py1, py2, short=True)

            # Draw COM if helper exists
            if self.com_helper is not None:
                frame = self.com_helper.drawFigure(frame, frame_idx)

            if show_preview:
                preview_frame = cv2.resize(frame, (self.frame_width // 2, self.frame_height // 2))
                cv2.imshow(f"{'Side1' if is_side1 else 'Side2'} View", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            out.write(frame)
            processed += 1

        out.release()
        if show_preview:
            cv2.destroyAllWindows()

        print(f"Finished processing video; Total Frames written: {processed}")
        print(f"========== SideVectorOverlay ({'Side1' if is_side1 else 'Side2'}) END ==========\n")
        
    # def ShortVectorOverlay(self, df_aligned, outputName=None, show_preview=True,
    #                       lag=0, com_csv_path=None, show_landmarks=False,
    #                       boundary_start=0, boundary_end=None):
    #     """
    #     Short view vector overlay using df_aligned for exact frame/force mapping.

    #     Matches your SHORT VIEW math:
    #     - scaling like: self.normalizeForces([0], self.fx2, [0], self.fz2)
    #         => scale factor is computed ONLY from Plate 2 Fx and Plate 2 Fz
    #     - if plate 2 is in front:
    #             fx1 = -Fx1, fx2 = -Fx2, fy1 = Fz1, fy2 = Fz2
    #             px unchanged, py flipped (1 - py)
    #         else (plate 1 in front):
    #             fx1 = +Fx1, fx2 = +Fx2, fy1 = Fz1, fy2 = Fz2
    #             px flipped (1 - px), py flipped (1 - py)

    #     Uses your trapezoid mapping via drawArrows(..., short=True).
    #     """
    #     # Load COM data
    #     # self.load_com_data(com_csv_path)
    #     self.load_com_helper(com_csv_path)

    #     if boundary_end is None:
    #         boundary_end = self.frame_count - 1

    #     print("\n========== ShortVectorOverlay (df_aligned) START ==========")
    #     print(f"df_aligned rows: {len(df_aligned)}")
    #     print(f"Video frames: {self.frame_count}, fps: {self.fps}")

    #     if self.frame_width is None or self.frame_height is None:
    #         print("Error: Frame data not set.")
    #         return

    #     if not outputName:
    #         outputName = "short_view_overlay_output.mp4"
    #         print(f"No output name provided, using default: {outputName}")

    #     # ---- Determine which plate is in front ONCE ----
    #     if len(self.corners) < 8:
    #         print("Error: corners not set. Run check_corner(view=...) first.")
    #         return

    #     plate2_in_front = self.corners[0][1] < self.corners[4][1]
    #     print(f"Force plate {'2' if plate2_in_front else '1'} is in front (short view)")

    #     # ---- 1) Compute scale factor like normalizeForces([0], fx2, [0], fz2) ----
    #     # i.e., ONLY Plate 2 Fx and Plate 2 Fz drive the max_force
    #     # --- same as GUI: normalizeForces([0], self.fx2, [0], self.fz2) ---
    #     scale_factor = self.scale_factor(
    #         [0.0],
    #         df_aligned["FP2_Fx"].astype(float).fillna(0.0).to_numpy(),
    #         [0.0],
    #         df_aligned["FP2_Fz"].astype(float).fillna(0.0).to_numpy(),
    #     )

    #     self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #     out = cv.VideoWriter(
    #         outputName,
    #         cv.VideoWriter_fourcc(*'mp4v'),
    #         self.fps,
    #         (self.frame_width, self.frame_height)
    #     )

    #     # Optional: force time column for debug
    #     if "Time(s)" in df_aligned.columns:
    #         force_time_array = df_aligned["Time(s)"].astype(float).to_numpy()
    #         time_col_name = "Time(s)"
    #     elif "abs time (s)" in df_aligned.columns:
    #         force_time_array = df_aligned["abs time (s)"].astype(float).to_numpy()
    #         time_col_name = "abs time (s)"
    #     else:
    #         force_time_array = None
    #         time_col_name = None

    #     processed = 0
    #     com_drawn_count = 0

    #     for idx, row in df_aligned.iterrows():
    #         frame_idx = int(row["FrameNumber"])
    #         if frame_idx < 0 or frame_idx >= self.frame_count:
    #             print(f"[WARN] Row {idx}: FrameNumber {frame_idx} out of range, skipping.")
    #             continue

    #         if frame_idx < boundary_start or frame_idx > boundary_end:
    #             continue

    #         self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    #         ret, frame = self.video.read()
    #         if not ret:
    #             print(f"[WARN] Could not read frame {frame_idx}, stopping.")
    #             break

    #         # --- raw forces from df_aligned ---
    #         Fx1 = float(row.get("FP1_Fx", 0.0) or 0.0)
    #         Fz1 = float(row.get("FP1_Fz", 0.0) or 0.0)
    #         Fx2 = float(row.get("FP2_Fx", 0.0) or 0.0)
    #         Fz2 = float(row.get("FP2_Fz", 0.0) or 0.0)

    #         # --- raw pressures from df_aligned ---
    #         Ax1 = float(row.get("FP1_Ax", 0.0) or 0.0)
    #         Ay1 = float(row.get("FP1_Ay", 0.0) or 0.0)
    #         Ax2 = float(row.get("FP2_Ax", 0.0) or 0.0)
    #         Ay2 = float(row.get("FP2_Ay", 0.0) or 0.0)

    #         # --- same normalization as GUI readData() ---
    #         px1 = np.clip((Ax1 + 0.3) / 0.6, 0, 1)
    #         py1 = np.clip((Ay1 + 0.45) / 0.9, 0, 1)
    #         px2 = np.clip((Ax2 + 0.3) / 0.6, 0, 1)
    #         py2 = np.clip((Ay2 + 0.45) / 0.9, 0, 1)

    #         # --- same short-view force & flip logic as GUI ---
    #         if plate2_in_front:
    #             fx1 = -Fx1 * scale_factor
    #             fx2 = -Fx2 * scale_factor
    #             fy1 =  Fz1 * scale_factor
    #             fy2 =  Fz2 * scale_factor
    #             py1 = 1 - py1
    #             py2 = 1 - py2
    #         else:
    #             fx1 =  Fx1 * scale_factor
    #             fx2 =  Fx2 * scale_factor
    #             fy1 =  Fz1 * scale_factor
    #             fy2 =  Fz2 * scale_factor
    #             px1 = 1 - px1
    #             px2 = 1 - px2
    #             py1 = 1 - py1
    #             py2 = 1 - py2
            
    #         if processed < 10 or processed % 30 == 0:
    #             video_t = frame_idx / self.fps if self.fps else 0.0
    #             if force_time_array is not None and 0 <= idx < len(force_time_array):
    #                 force_t = force_time_array[idx]
    #                 print(
    #                     f"[DEBUG] row={idx:5d}, frame={frame_idx:5d}, "
    #                     f"video_t={video_t:.4f}s, {time_col_name}={force_t:.4f}s, "
    #                     f"Δ={force_t - video_t:.4f}s, "
    #                 )
    #             else:
    #                 print(
    #                     f"[DEBUG] row={idx:5d}, frame={frame_idx:5d}, "
    #                     f"video_t={video_t:.4f}s, "
    #                 )

    #         self.drawArrows(frame, fx1, fx2, fy1, fy2, px1, px2, py1, py2, short=True)

    #          # ----- Optionally draw landmarks -----
    #         debug_com = (processed < 10)
    #         # frame = self.draw_com_on_frame(frame, frame_idx, debug=debug_com)
    #         if self.com_helper is not None:
    #             frame = self.com_helper.drawFigure(frame, frame_idx)

    #         # Track how many frames had COM drawn
    #         if self.com_data is not None and frame_idx in self.com_data.index:
    #             com_drawn_count += 1

    #         # Show preview if desired
    #         if show_preview:
    #             preview_frame = cv2.resize(frame, (self.frame_width // 2, self.frame_height // 2))
    #             cv2.imshow("Short View with COM", preview_frame)
    #             if cv2.waitKey(1) & 0xFF == ord("q"):
    #                 break

    #         out.write(frame)
    #         processed += 1

    #     out.release()
    #     if show_preview:
    #         cv2.destroyAllWindows()

    #     print(f"Processed {processed} frames, COM drawn on {com_drawn_count} frames")
    #     print("========== ShortVectorOverlay (df_aligned) END ==========\n")

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