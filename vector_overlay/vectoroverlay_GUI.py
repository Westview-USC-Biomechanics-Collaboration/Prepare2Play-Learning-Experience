# Import libraries
import cv2
import cv2 as cv
from matplotlib import lines
import pandas as pd
from vector_overlay.select_corners import select_points
import numpy as np
import os

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
        rename_dict = {
            "Fx": "Fx1", "Fy": "Fy1", "Fz": "Fz1", "|Ft|": "Ft1", "Ax": "Ax1", "Ay": "Ay1",
            "Fx.1": "Fx2", "Fy.1": "Fy2", "Fz.1": "Fz2", "|Ft|.1": "Ft2", "Ax.1": "Ax2", "Ay.1": "Ay2",
            "Fx.2": "Fx3", "Fy.2": "Fy3", "Fz.2": "Fz3", "|Ft|.2": "Ft3", "Ax.2": "Ax3", "Ay.2": "Ay3"
        }

        for key in rename_dict:
            if key in data.columns:
                data.rename(columns={key: rename_dict[key]}, inplace=True)

        self.data = data
        self.video = video
        self.force_fps = force_fps

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

        # Initialize
        self.setFrameData()
        self.check_corner(cap=self.video)
        self.readData()

    def check_corner(self, cap):
        self.corners = select_points(cap=cap, num_points=8)

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
        """Improved data reading with better synchronization"""
        force_samples = len(self.data)
        video_frames = self.frame_count
        video_duration = video_frames / self.fps




        print(f"Force data: {force_samples}")
        print(f"Video: {video_frames} frames at {self.fps} fps ({video_duration:.2f}s)")

        # Calculate samples per frame with offset

        #Alternatively, formula = num of colums (240 * time_elapsed)
        total_time_elapsed = self.data["abs time (s)"].iloc[-1] * 240

        #Get total # of rows

        #total_rows = len(self.data) #Not working with VSCode reading


        samples_per_frame = 10  # Default to 5 samples per frame for 1200 Hz data, 10 samples per frame for 2400 hz data


        print(f"# of total samples: {len(self.data)}")

        print(f"Samples per frame: {samples_per_frame:.3f}")
        #print(f"Time offset: {self.time_offset}s ({offset_samples} samples)")

        # Initialize arrays
        fx1, fy1, fz1, px1, py1 = [], [], [], [], []
        fx2, fy2, fz2, px2, py2 = [], [], [], [], []

        # Extract timestamps from DataFrame


        for frame_idx in range(video_frames):
            # Calculate corresponding data index with offset
            frame_time = frame_idx / (self.fps)
            # Use stepsize to sample force data for each frame
            stepsize = max(1, int(force_samples / video_frames)) if video_frames > 0 else 1
            data_idx = frame_idx * stepsize
            print(f"Data index: {data_idx}")

            # Ensure index is within bounds
            if 0 <= data_idx < len(self.data):
                row = self.data.iloc[data_idx]

                # Extract data with better error handling
                data_x1 = row.get("Fx1", 0.0) if not pd.isna(row.get("Fx1")) else 0.0
                data_y1 = row.get("Fy1", 0.0) if not pd.isna(row.get("Fy1")) else 0.0
                data_z1 = row.get("Fz1", 0.0) if not pd.isna(row.get("Fz1")) else 0.0

                # Normalize pressure coordinates (0-1 range)
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
                # Default values for out-of-bounds indices
                data_x1 = data_y1 = data_z1 = 0.0
                pressure_x1 = pressure_y1 = 0.5  # Center position
                data_x2 = data_y2 = data_z2 = 0.0
                pressure_x2 = pressure_y2 = 0.5

            # Append to arrays
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

        # Convert to tuples
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

    def LongVectorOverlay(self, outputName=None, show_preview=True, lag=0):
        """Long view vector overlay with lag-based video/data alignment (video starts earlier if lag < 0)"""
        print("Lag parameter:", lag)
        # If lag is in frames, convert to seconds (if user passes a large int)
        lag_seconds = lag / self.fps
        print(f"Applying lag of {lag_seconds} seconds to force data synchronization...")
        self.normalizeForces(self.fy1, self.fy2, self.fz1, self.fz2)

        if self.frame_width is None or self.frame_height is None:
            print("Error: Frame data not set.")
            return

        if not outputName:
            outputName = "long_view_overlay_output.mp4"
            print(f"No output name provided, using default: {outputName}")

        # Reset video to beginning
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_number = 0
        print("Starting long view overlay processing...")

        # If lag is positive, skip video frames (video starts later)
        # If lag is negative, skip force data samples (video starts earlier)
        frames_to_skip = int(abs(lag_seconds) * self.fps)
        force_idx_offset = 0
        if lag_seconds > 0:
            print(f"Skipping {frames_to_skip} video frames to align video with force data.")
            for _ in range(frames_to_skip):
                self.video.read()
        elif lag_seconds < 0:
            print(f"Skipping {frames_to_skip} force data samples to start video earlier.")
            force_idx_offset = frames_to_skip

        out = cv.VideoWriter(outputName, cv.VideoWriter_fourcc(*'mp4v'), self.fps,
                            (self.frame_width, self.frame_height))

        # Only process the frames that have corresponding force data
        while self.video.isOpened() and frame_number + force_idx_offset < len(self.fx1):
            ret, frame = self.video.read()
            if not ret:
                print(f"Can't read frame at position {frame_number}")
                break

            force_idx = frame_number + force_idx_offset

            fx1 = -self.fy1[force_idx]
            fx2 = -self.fy2[force_idx]
            fy1 = self.fz1[force_idx]
            fy2 = self.fz2[force_idx]
            px1 = self.py1[force_idx]
            py1 = self.px1[force_idx]
            px2 = self.py2[force_idx]
            py2 = self.px2[force_idx]

            # Debugging output for force values
            print(f"Frame {frame_number}: fx1={fx1}, fy1={fy1}, px1={px1}, py1={py1}")


            self.drawArrows(frame, fx1, fx2, fy1, fy2, px1, px2, py1, py2)

            if show_preview:
                cv2.imshow("Long View", cv2.resize(frame, (int(self.frame_width * 0.5), int(self.frame_height * 0.5))))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_number += 1
            out.write(frame)

            if frame_number % 30 == 0:
                print(f"Processed {frame_number}/{len(self.fx1) - force_idx_offset} frames")

        out.release()
        if show_preview:
            cv2.destroyAllWindows()
        print(f"Finished processing video; Total Frames: {frame_number}")

    def TopVectorOverlay(self, outputName=None, show_preview=True):
        """Top view vector overlay with optimized processing"""
        self.normalizeForces(self.fy1, self.fy2, self.fx1, self.fx2)

        if self.frame_width is None or self.frame_height is None:
            print("Error: Frame data not set.")
            return

        # Handle None or empty output name
        if not outputName:
            outputName = "top_view_overlay_output.mp4"
            print(f"No output name provided, using default: {outputName}")

        # Reset video to beginning
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        out = cv.VideoWriter(outputName, cv.VideoWriter_fourcc(*'mp4v'), self.fps,
                           (self.frame_width, self.frame_height))

        frame_number = 0

        print("Starting top view overlay processing...")

        while self.video.isOpened() and frame_number < len(self.fx1):
            ret, frame = self.video.read()
            if not ret:
                print(f"Can't read frame at position {frame_number}")
                break

            # Map forces to view coordinates
            fx1 = -self.fy1[frame_number]  # -Fy maps to x in top view
            fx2 = -self.fy2[frame_number]
            fy1 = -self.fx1[frame_number]  # -Fx maps to y in top view
            fy2 = -self.fx2[frame_number]

            # Map pressure positions
            px1 = self.py1[frame_number]
            py1 = 1 - self.px1[frame_number]  # Invert y-coordinate
            px2 = self.py2[frame_number]
            py2 = 1 - self.px2[frame_number]

            self.drawArrows(frame, fx1, fx2, fy1, fy2, px1, px2, py1, py2)

            if show_preview:
                cv2.imshow("Top View", cv2.resize(frame, (int(self.frame_width * 0.5), int(self.frame_height * 0.5))))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_number += 1
            out.write(frame)

            # Progress indicator
            if frame_number % 30 == 0:
                print(f"Processed {frame_number}/{len(self.fx1)} frames")

        out.release()
        if show_preview:
            cv2.destroyAllWindows()
        print(f"Finished processing video; Total Frames: {frame_number}")

    def ShortVectorOverlay(self, outputName=None, show_preview=True):
        """Short view vector overlay with optimized processing"""
        self.normalizeForces([0], self.fx2, [0], self.fz2)

        if self.frame_width is None or self.frame_height is None:
            print("Error: Frame data not set.")
            return

        # Handle None or empty output name
        if not outputName:
            outputName = "short_view_overlay_output.mp4"
            print(f"No output name provided, using default: {outputName}")

        # Reset video to beginning
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        out = cv.VideoWriter(outputName, cv.VideoWriter_fourcc(*'mp4v'), self.fps,
                           (self.frame_width, self.frame_height))

        frame_number = 0

        print("Starting short view overlay processing...")

        # Determine which plate is in front (only print once)
        plate2_in_front = self.corners[0][1] < self.corners[4][1]
        print(f"Force plate {'2' if plate2_in_front else '1'} is in front")

        while self.video.isOpened() and frame_number < len(self.fx1):
            ret, frame = self.video.read()
            if not ret:
                print(f"Can't read frame at position {frame_number}")
                break

            if plate2_in_front:
                # Plate 2 in front
                fx1 = -self.fx1[frame_number]
                fx2 = -self.fx2[frame_number]
                fy1 = self.fz1[frame_number]
                fy2 = self.fz2[frame_number]
                px1 = self.px1[frame_number]
                px2 = self.px2[frame_number]
                py1 = 1 - self.py1[frame_number]
                py2 = 1 - self.py2[frame_number]
            else:
                # Plate 1 in front
                fx1 = self.fx1[frame_number]
                fx2 = self.fx2[frame_number]
                fy1 = self.fz1[frame_number]
                fy2 = self.fz2[frame_number]
                px1 = 1 - self.px1[frame_number]
                px2 = 1 - self.px2[frame_number]
                py1 = 1 - self.py1[frame_number]
                py2 = 1 - self.py2[frame_number]

            self.drawArrows(frame, fx1, fx2, fy1, fy2, px1, px2, py1, py2, short=True)

            if show_preview:
                cv2.imshow("Short View", cv2.resize(frame, (int(self.frame_width * 0.5), int(self.frame_height * 0.5))))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_number += 1
            out.write(frame)

            # Progress indicator
            if frame_number % 30 == 0:
                print(f"Processed {frame_number}/{len(self.fx1)} frames")

        out.release()
        if show_preview:
            cv2.destroyAllWindows()
        print(f"Finished processing video; Total Frames: {frame_number}")

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