import traceback
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from vector_overlay.Cal_COM import calculateCOM
import csv
import time
import threading
import multiprocessing as processing
from multiprocessing import Process, Queue, Pipe
import queue
import cv2 as cv
from vector_overlay.select_corners import select_points
import os
from Util.COM_helper import COM_helper



"""""
This is the skeleton overlay/stick figure
You need to scroll down to the last line an change the value of video_path in order to process the stick figure
"""""

def find_coordinates(sex, video_path, filename, confidencelevel=0.85, displayname=False, displaystickfigure=False,
                        displayCOM=False):
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=confidencelevel, model_complexity=2)

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define the codec and create VideoWriter object to save the annotated video
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame to find pose landmarks
            results = pose.process(frame)
            pose_landmarks_list = [results.pose_landmarks] if results.pose_landmarks else []

            # Draw landmarks on the frame
            annotated_frame = draw_landmarks_on_image(np.copy(frame), pose_landmarks_list, sex, displayname,
                                                    displaystickfigure, displayCOM)

            # Write the annotated frame to the output video file
            out.write(annotated_frame)

            # Display the annotated frame (optional)
            cv2.imshow('Annotated Frame', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video capture and writer objects
        cap.release()
        out.release()
        cv2.destroyAllWindows()


def draw_landmarks_on_image(annotated_image, pose_landmarks_list, sex, displayname=False, displaystickfigure=False,
                            displayCOM=False):
    pose_landmark_names = {
        0: "nose",
        1: "left_eye_inner",
        2: "left_eye",
        3: "left_eye_outer",
        4: "right_eye_inner",
        5: "right_eye",
        6: "right_eye_outer",
        7: "left_ear",
        8: "right_ear",
        9: "mouth_left",
        10: "mouth_right",
        11: "LSHOULDER",
        12: "RSHOULDER",
        13: "LELBOW",
        14: "RELBOW",
        15: "LWRIST",
        16: "RWRIST",
        17: "left_pinky",
        18: "right_pinky",
        19: "left_index",
        20: "right_index",
        21: "left_thumb",
        22: "right_thumb",
        23: "LHIP",
        24: "RHIP",
        25: "LKNEE",
        26: "RKNEE",
        27: "LANKLE",
        28: "RANKLE",
        29: "LHEEL",
        30: "RHEEL",
        31: "LTOE",
        32: "RTOE"
    }

    # Define connections between landmarks
    pose_connections = mp.solutions.pose.POSE_CONNECTIONS

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        if not pose_landmarks:
            continue

        data = []
        # Draw circles at each landmark position
        for id, landmark in enumerate(pose_landmarks.landmark):
            landmark_name = pose_landmark_names.get(id, f"landmark_{id}")

            # Only draw visible landmarks
            if landmark.visibility > 0:
                h, w, _ = annotated_image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                data.append(cx)
                data.append(cy)
                if displaystickfigure == True:
                    # Draw landmark as a circle
                    cv2.circle(annotated_image, (cx, cy), 5, (0, 255, 0), -1)
                    # Draw lines connecting landmarks
                    for connection in pose_connections:
                        start_landmark = pose_landmarks.landmark[connection[0]]
                        end_landmark = pose_landmarks.landmark[connection[1]]

                        # Only draw the line if both landmarks are visible
                        if start_landmark.visibility > 0 and end_landmark.visibility > 0:
                            start_x, start_y = int(start_landmark.x * w), int(start_landmark.y * h)
                            end_x, end_y = int(end_landmark.x * w), int(end_landmark.y * h)

                            # Draw the line between the two landmarks
                            cv2.line(annotated_image, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)

                    # optional: Put text next to the joint
                    if displayname:
                        cv2.putText(annotated_image, landmark_name, (cx + 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1, cv2.LINE_AA)

        # Get index label
        columns_name = []
        for i in range(len(pose_landmark_names)):
            columns_name.append(str(pose_landmark_names[i]) + "_x")
            columns_name.append(str(pose_landmark_names[i]) + "_y")

        # Find COM
        if displayCOM == True:
            datain = pd.Series(data, index=columns_name, name="Datain Series")
            dataout = calculateCOM(datain, sex)
            cv2.circle(annotated_image, (int(dataout[0]), int(dataout[1])), 12, (0, 0, 255), -1)

    return annotated_image


def pose_landmarks_to_row(pose_landmarks_list):
    """
    Convert pose_landmarks_list (list of 33 [x,y,z]) or empty list
    to dict with keys landmark_i_x, landmark_i_y, landmark_i_visibility.
    If input is empty or invalid, return row filled with 0.0.
    """

    num_landmarks = 33
    row = {}

    # Initialize with zeros
    for i in range(num_landmarks):
        row[f"landmark_{i}_x"] = 0.0
        row[f"landmark_{i}_y"] = 0.0
        row[f"landmark_{i}_visibility"] = 0.0

    # If no landmarks detected, just return zeros row
    if not pose_landmarks_list or len(pose_landmarks_list) != num_landmarks:
        return row

    # For each landmark, assign values
    for i, landmark in enumerate(pose_landmarks_list):
        # landmark is a list or tuple: [x, y, z]
        if isinstance(landmark, (list, tuple)) and len(landmark) == 3:
            x, y, z = landmark
            row[f"landmark_{i}_x"] = float(x)
            row[f"landmark_{i}_y"] = float(y)
            # Set visibility 1.0 if any coord is nonzero, else 0.0
            if x != 0.0 or y != 0.0 or z != 0.0:
                row[f"landmark_{i}_visibility"] = 1.0
            else:
                row[f"landmark_{i}_visibility"] = 0.0
        else:
            # If bad data, keep zeros
            pass

    return row

def process_frame(q: processing.Queue, resultsVector_queue: processing.Queue, resultsCOM_queue: processing.Queue, cfg):
    import mediapipe as mp
    import cv2
    import pandas as pd
    from vector_overlay.Cal_COM import calculateCOM # Import calculateCOM here as well

    sex = cfg['sex']
    confidencelevel = cfg['confidence']
    displayCOM = cfg['displayCOM']

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

    def drawArrows(frame, xf1, xf2, yf1, yf2, px1, px2, py1, py2, corners, short=False):
        """Draw force arrows on frame"""
        if short:
            point_pair1 = rect_to_trapezoid(px1, py1, 1, 1,
                                           [corners[0], corners[1], corners[2], corners[3]], short=True)
            point_pair2 = rect_to_trapezoid(px2, py2, 1, 1,
                                           [corners[4], corners[5], corners[6], corners[7]], short=True)
        else:
            point_pair1 = rect_to_trapezoid(px1, py1, 1, 1,
                                           [corners[0], corners[1], corners[2], corners[3]])
            point_pair2 = rect_to_trapezoid(px2, py2, 1, 1,
                                           [corners[4], corners[5], corners[6], corners[7]])

        end_point_1 = (int(point_pair1[0] + xf1), int(point_pair1[1] - yf1))
        end_point_2 = (int(point_pair2[0] + xf2), int(point_pair2[1] - yf2))

        # Draw arrows with different colors for each plate
        cv.arrowedLine(frame, point_pair1, end_point_1, (0, 255, 0), 4)  # Green for plate 1
        cv.arrowedLine(frame, point_pair2, end_point_2, (255, 0, 0), 4)  # Blue for plate 2

    pose_landmark_names = {
        0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
        4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer", 7: "left_ear",
        8: "right_ear", 9: "mouth_left", 10: "mouth_right", 11: "LSHOULDER",
        12: "RSHOULDER", 13: "LELBOW", 14: "RELBOW", 15: "LWRIST", 16: "RWRIST",
        17: "left_pinky", 18: "right_pinky", 19: "left_index", 20: "right_index",
        21: "left_thumb", 22: "right_thumb", 23: "LHIP", 24: "RHIP", 25: "LKNEE",
        26: "RKNEE", 27: "LANKLE", 28: "RANKLE", 29: "LHEEL", 30: "RHEEL", 31: "LTOE", 32: "RTOE"
    }

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=confidencelevel, model_complexity=2) as pose:

        print("[INFO] Start processing COM")
        print(f"[PROCESS {processing.current_process().name}] Started.")
        while True:
            #print("[DEBUG] [COM] - Frame index:", frame_index)
            
            try:
                item = q.get()
            except queue.Empty:
                print(f"[PROCESS {processing.current_process().name}] Timed out waiting for frame.")
                resultsCOM_queue.put(None)
                break

            if item is None:
                print(f"[PROCESS {processing.current_process().name}] got sentinel and is exiting.")
                resultsCOM_queue.put(None)
                resultsVector_queue.put(None)
                break  # Sentinel to signal "no more data"
            # print(f"[PROCESS {processing.current_process().name}] Got frame {item[0]}")

            frame_index, frame_data = item
            try:
                # print("[DEBUG] [COM] - Processing frame index:", frame_index)
                # Ensure frame data is not None before processing
                if frame_data is None:
                    print(f"[PROCESS {processing.current_process().name}] Received None frame data for index {frame_index}. Skipping.")
                    resultsCOM_queue.put(None)
                    resultsVector_queue.put((frame_index, frame_data))
                    continue
                
                full_frame = frame_data
                frame = cv2.resize(frame_data, (0, 0), fx=0.3, fy=0.3)
                try:
                    results = pose.process(frame)
                    if results.pose_landmarks:
                        pose_landmarks_list = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]
                    else:
                        pose_landmarks_list = [[0.0, 0.0, 0.0]] * 33
                except Exception as e:
                    print(f"[ERROR] pose.process() failed: {e}")
                    pose_landmarks_list = [[0.0, 0.0, 0.0]] * 33

                if pose_landmarks_list is None:
                    print(f"[PROCESS {processing.current_process().name}] Timed out on frame {frame_index}")
                    pose_landmarks_list = [[0.0, 0.0, 0.0]] * 33  # so the CSV has a row
                    resultsCOM_queue.put(None)
                    continue

                row = pose_landmarks_to_row(pose_landmarks_list)

                # Add frame index to row
                row["frame_index"] = frame_index
                if displayCOM:
                    # Calculate COM and add to row
                    datain = pd.Series([row[f"landmark_{i}_x"] for i in range(33)] +
                                        [row[f"landmark_{i}_y"] for i in range(33)],
                                        index=[f"{pose_landmark_names[i]}_x" for i in range(33)] +
                                            [f"{pose_landmark_names[i]}_y" for i in range(33)])
                    dataout = calculateCOM(datain,sex)
                    row["COM_x"] = dataout[0]
                    row["COM_y"] = dataout[1]

                resultsCOM_queue.put(row)
                #print(f"Size of result_queue: {results_queue.qsize()}")

                fps = cfg['fps'] 
                lag = cfg['lag']
                frame_count = cfg['frame_count']
                frames_to_skip = int(abs(lag))
                force_offset = 0

                video_limit = frame_count - (frames_to_skip if lag > 0 else 0)
                data_limit  = len(cfg['fx1']) - (frames_to_skip if lag < 0 else 0)
                max_frames  = max(0, min(video_limit, data_limit))
                # if lag < 0:
                #     if frames_to_skip > max_frames:
                #         force_offset = max_frames
                #     else:
                #         force_offset = frames_to_skip
                # else:
                #     force_offset = 0
                # f_idx = frame_index + force_offset 

                # print(f"frame_index: {frame_index}, force_offset: {force_offset}, f_idx: {f_idx}")
                f_idx = frame_index
                # if 0 <= f_idx < len(cfg['fx1']):

                # Safety guard
                if not (0 <= f_idx < len(cfg['fx1'])):
                    print(f"[WORKER] frame_index {frame_index} has out-of-bounds f_idx={f_idx}")
                    resultsVector_queue.put((frame_index, full_frame))
                    resultsCOM_queue.put(row)
                    continue
                _fx1 = -cfg['fy1'][f_idx] 
                _fx2 = -cfg['fy2'][f_idx]
                _fy1 =  cfg['fz1'][f_idx] 
                _fy2 =  cfg['fz2'][f_idx]
                _px1 =  cfg['py1'][f_idx] 
                _py1 =  cfg['px1'][f_idx]
                _px2 =  cfg['py2'][f_idx] 
                _py2 =  cfg['px2'][f_idx]
                if frame_index < 10:
                    print(f"[WORKER] frame={frame_index}, f_idx={f_idx}, "
                        f"fx1={_fx1:.2f}, fy1={_fy1:.2f}, "
                        f"fx2={_fx2:.2f}, fy2={_fy2:.2f}, "
                        f"px1={_px1:.2f}, py1={_py1:.2f}, "
                        f"px2={_px2:.2f}, py2={_py2:.2f}")
                drawArrows(full_frame, _fx1, _fx2, _fy1, _fy2, _px1, _px2, _py1, _py2, cfg['corners'])

                # --- sends to Writer Queue ---
                resultsVector_queue.put((frame_index, full_frame))   # for writer

            except Exception as e:
                print(f"Error processing frame {frame_index}: {e}")
                print(traceback.format_exc())
                # Still forward the original full_frame so writer can progress
                try:
                    resultsVector_queue.put((frame_index, full_frame))
                except Exception:
                    pass
                resultsCOM_queue.put(None)
                break

def frame_reader(frame_queue: processing.Queue, video_path, start_frame, num_workers, max_frames, cfg):
    print(f"[READER] Attempting to open video: '{video_path}'") # <-- ADD THIS LINE
    total_frames = cfg['frame_count']
    video_limit = total_frames - (start_frame if cfg['lag'] > 0 else 0)
    data_limit  = len(cfg['fx1']) - (start_frame if cfg['lag'] < 0 else 0)
    skip_max = max(0, min(video_limit, data_limit))
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    index = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    print("Currently reading frame:", current_frame)
    print("[INFO] Loading frames from video...")
    while cap.isOpened() and index < total_frames:
        ret, frame = cap.read()
        if not ret:
            break 
        try:
            frame_queue.put((index, frame.copy()))  # <--- ADDED timeout
        except:
            print(f"[READER] Timeout trying to put frame {index}")
            break
        index += 1
    cap.release()
    print("[READER] Done reading video. Sending sentinels...")
    for _ in range(num_workers):
        frame_queue.put(None)
    print("[READER] Sent all sentinels.")

def writer_process(results_q, out_path, fps, frame_size, num_workers):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, frame_size)
    next_to_write, buffer, sentinels = 0, {}, 0
    while True:
        if sentinels >= num_workers and not buffer:
            break
        try:
            item = results_q.get(timeout=5.0)
        except queue.Empty:
            continue
        if item is None:
            sentinels += 1
            continue
        idx, frame = item
        buffer[idx] = frame
        while next_to_write in buffer:
            out.write(buffer.pop(next_to_write))
            next_to_write += 1
    out.release()

class Processor:
    def __init__(self, video_path, data_df, lag, output_mp4, force_fps=None):
        self.video_path = video_path
        self.data = data_df
        self.lag = lag
        self.output_mp4 = output_mp4
        self.force_fps  = force_fps

        # open video to read metadata
        print("Opening:", self.video_path, os.path.exists(self.video_path))
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
        self.frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps          = int(cap.get(cv2.CAP_PROP_FPS))
        self.frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # vectors/pressures
        self.fx1=self.fy1=self.fz1=self.px1=self.py1=()
        self.fx2=self.fy2=self.fz2=self.px2=self.py2=()

        self.corners = []  

        self.readData()
        self.normalizeForces(self.fx1, self.fx2, self.fy1, self.fy2)
        self.debug_force_array_mapping(n=10)
        self.apply_lag_alignment()

    def check_corner(self, view):
        cap = cv2.VideoCapture(self.video_path)
        self.corners = select_points(self, cap=cap, view=view)

    def setFrameData(self):
        cap = cv2.VideoCapture(self.video_path)

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
            # print(f"Data index: {data_idx}")

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
    
    def debug_force_array_mapping(self, n=10):
        """
        Compare what we *think* we're putting in fx1/fx2... to the raw DataFrame values.
        This checks the stepsize/data_idx logic AND plate-1/plate-2 column mapping.
        """
        force_samples = len(self.data)
        video_frames = self.frame_count
        stepsize = max(1, int(force_samples / video_frames)) if video_frames > 0 else 1

        print("\n[DEBUG] Force array mapping check")
        print(f"force_samples={force_samples}, video_frames={video_frames}, stepsize={stepsize}")
        print("frame_idx | data_idx | Fx1_raw  Fx1_arr  Fx2_raw  Fx2_arr  Fz1_raw  Fz1_arr  Fz2_raw  Fz2_arr")

        for frame_idx in range(min(n, video_frames)):
            data_idx = frame_idx * stepsize
            if data_idx >= force_samples:
                break

            row = self.data.iloc[data_idx]
            Fx1_raw = row.get("Fx1", None)
            Fx2_raw = row.get("Fx2", None)
            Fz1_raw = row.get("Fz1", None)
            Fz2_raw = row.get("Fz2", None)

            Fx1_arr = self.fx1[frame_idx]
            Fx2_arr = self.fx2[frame_idx]
            Fz1_arr = self.fz1[frame_idx]
            Fz2_arr = self.fz2[frame_idx]

            print(f"{frame_idx:8d} | {data_idx:8d} | "
                f"{Fx1_raw:7.2f} {Fx1_arr:7.2f}  "
                f"{Fx2_raw:7.2f} {Fx2_arr:7.2f}  "
                f"{Fz1_raw:7.2f} {Fz1_arr:7.2f}  "
                f"{Fz2_raw:7.2f} {Fz2_arr:7.2f}")
    
    def apply_lag_alignment(self):
        """
        Align video vs force data once, so workers have 1:1 frame<->sample.
        Assume self.lag is in FRAMES (if it's seconds, convert with int(round(lag_seconds * self.fps))).
        """
        lag_seconds = self.lag / self.fps
        lag_frames = int(abs(lag_seconds) * self.fps)

        print(f"Lag: {self.lag}")
        print(f"lag_frames: {lag_frames}")

        video_limit = self.frame_count - (lag_frames if self.lag > 0 else 0)
        data_limit  = len(self.fx1) - (lag_frames if self.lag < 0 else 0)
        max_frames  = max(0, min(video_limit, data_limit))


        # Where to start the reader
        self.reader_start_frame = 0
        force_idx_offset = 0

        if lag_seconds > 0:
            # Video should start later -> advance video by lag_frames
            self.reader_start_frame = min(lag_frames, max(0, self.frame_count - 1))
            print(f"reader start_frame: {self.reader_start_frame}")
        elif lag_seconds < 0:
            if lag_frames > max_frames:
                force_idx_offset = max_frames
                print(f"Skipping {force_idx_offset} force data samples to start video earlier.")
            else:
                print(f"Skipping {lag_frames} force data samples to start video earlier.")
                force_idx_offset = lag_frames
            # Video should start earlier -> trim force arrays by -lag_frames
            def trim(t, force_idx_offset):
                # safe trim when arrays may be shorter than off
                return t[force_idx_offset:]

            self.fx1 = trim(self.fx1, force_idx_offset) 
            self.fx2 = trim(self.fx2, force_idx_offset)
            self.fy1 = trim(self.fy1, force_idx_offset) 
            self.fy2 = trim(self.fy2, force_idx_offset)
            self.fz1 = trim(self.fz1, force_idx_offset) 
            self.fz2 = trim(self.fz2, force_idx_offset)
            self.px1 = trim(self.px1, force_idx_offset) 
            self.px2 = trim(self.px2, force_idx_offset)
            self.py1 = trim(self.py1, force_idx_offset) 
            self.py2 = trim(self.py2, force_idx_offset)

            # after trimming, set lag to 0 so downstream code doesn’t re-apply it
            self.lag = 0

        # Also cap the effective length so video and force arrays have overlap
        max_len = min(
            self.frame_count - self.reader_start_frame, len(self.fx1)
        )
        # Optionally trim all arrays to max_len to keep things simple:
        # self.fx1 = self.fx1[:max_len]; self.fx2 = self.fx2[:max_len]
        # self.fy1 = self.fy1[:max_len]; self.fy2 = self.fy2[:max_len]
        # self.fz1 = self.fz1[:max_len]; self.fz2 = self.fz2[:max_len]
        # self.px1 = self.px1[:max_len]; self.px2 = self.px2[:max_len]
        # self.py1 = self.py1[:max_len]; self.py2 = self.py2[:max_len]
        self.effective_frames = max_len
    
    def SaveToTxt(self, sex, filename, confidencelevel=0.85, displayCOM=False):
        startTime = time.time()

        cfg = dict(
            sex=sex, confidence=confidencelevel, displayCOM=displayCOM,
            fps=self.fps, frame_count=self.frame_count, frame_width=self.frame_width, frame_height=self.frame_height, 
            lag=self.lag, corners=self.corners,
            fx1=self.fx1, fx2=self.fx2, fy1=self.fy1, fy2=self.fy2,
            fz1=self.fz1, fz2=self.fz2, px1=self.px1, px2=self.px2,
            py1=self.py1, py2=self.py2
        )

        frame_queue = Queue()  
        resultCOM_queue = Queue()
        resultVector_queue = Queue()

        num_workers = min(6, processing.cpu_count())

        reader_thread = threading.Thread(target=frame_reader, args=(frame_queue, self.video_path, self.reader_start_frame, num_workers, self.effective_frames, cfg))
        reader_thread.start()

        writer = processing.Process(
            target=writer_process,
            args=(resultVector_queue, self.output_mp4, self.fps, (self.frame_width, self.frame_height), num_workers),
            daemon=True
        )
        writer.start()


        # Start worker processes
        workers: list[Process] = []
        for i in range(num_workers):
            print(f"[PROCESS] Starting process {i}")
            p = processing.Process(target=process_frame,
                       args=(frame_queue, resultVector_queue, resultCOM_queue, cfg),
                       daemon=True)
            p.start()
            workers.append(p)

        reader_thread.join()
        print("[MAIN] Reader thread finished")

        print(f"Size of result_queue: {resultCOM_queue.qsize()}")
        # Drain result queue
        output = []
        sentinel_count = 0
        while sentinel_count < num_workers:
            try:
                item = resultCOM_queue.get() # Add a timeout here for debugging
                if item is None:
                    sentinel_count += 1
                    print(f"[MAIN] Received sentinel from worker ({sentinel_count}/{num_workers})")
                else:
                    output.append(item)
            except Exception as e:
                print(f"[MAIN ERROR] Timeout or error while draining result queue: {e}")
                print(f"[MAIN ERROR] Current sentinel count: {sentinel_count}/{num_workers}")
                print(f"[MAIN ERROR] Remaining workers not yet joined: {[p.name for p in workers if p.is_alive()]}")
                break # Break the loop if we timeout
        print(f"[MAIN] Drained all results from queue. Total items: {len(output)}")

        print("[MAIN] Waiting for all worker processes to finish joining...")
        joined_workers_count = 0
        for p in workers:
            print(f"[MAIN] Attempting to join worker {p.name} (PID: {p.pid})...")
            p.join() # Try to join with a timeout

            if p.is_alive():
                print(f"[MAIN ERROR] Worker {p.name} (PID: {p.pid}) did NOT join after timeout (still alive). Forcibly terminating...")
                p.terminate() # <-- Forcefully terminate the rogue process
                p.join() # Give it a moment to terminate after being signaled
                if p.is_alive():
                    print(f"[MAIN ERROR] Worker {p.name} (PID: {p.pid}) *still* alive after forceful termination!")
                else:
                    print(f"[MAIN] Worker {p.name} (PID: {p.pid}) successfully terminated after force.")
                    joined_workers_count += 1 # Count it as "handled" for joining purposes
            else:
                # Process successfully joined within the timeout
                joined_workers_count += 1
                print(f"[MAIN] Worker {p.name} (PID: {p.pid}) finished joining.")

                
        writer.join()

        output.sort(key=lambda x: x["frame_index"])
        df = pd.DataFrame(output)
        df.to_csv(filename, index=False)

        endTime = time.time()
        print("[INFO] COM Total time taken: ", endTime - startTime)

        count = 0
        vector_cam = cv2.VideoCapture(self.output_mp4)
        com_helper = COM_helper()
        while(vector_cam.isOpened() and count < self.effective_frames):
            ret, frame = vector_cam.read()
            frame = com_helper.drawFigure(frame, count)
            cv2.imshow("Long View", cv2.resize(frame, (int(self.frame_width * 0.5), int(self.frame_height * 0.5))))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            count += 1

        vector_cam.release()
        cv2.destroyAllWindows()
        



# if __name__ == "__main__":
#     video_path = r"C:\Users\Student\Downloads\spu_lr_NS_long_vid01.mov"  # Change this to your video file path
#     #cap = cv2.VideoCapture(video_path)
#     print(f"DEBUG IN MAIN: Type of video_path before Processor: {type(video_path)}") # Add this
#     print(f"DEBUG IN MAIN: Value of video_path before Processor: '{video_path}'") # Add this

#     processor = Processor(video_path)

#     # Example usage:
#     # Draw skeleton with display options:
#     # processor.find_coordinates(sex='male', video_path, filename='output.mp4', displayname=True, displaystickfigure=True, displayCOM=True)

#     # Save landmarks to CSV with zeros for no detection
#     processor.SaveToTxt(sex='male', filename='pose_landmarks.csv', confidencelevel=0.85, displayCOM=True)