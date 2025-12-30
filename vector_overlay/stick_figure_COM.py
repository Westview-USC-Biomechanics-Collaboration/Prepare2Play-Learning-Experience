import traceback
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')
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

def process_frame(q: processing.Queue, results_queue: processing.Queue, sex, confidencelevel, displayCOM):
    import mediapipe as mp
    import cv2
    import pandas as pd
    from vector_overlay.Cal_COM import calculateCOM # Import calculateCOM here as well

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
                results_queue.put(None)
                break

            if item is None:
                print(f"[PROCESS {processing.current_process().name}] got sentinel and is exiting.")
                results_queue.put(None)
                break  # Sentinel to signal "no more data"
            # print(f"[PROCESS {processing.current_process().name}] Got frame {item[0]}")

            frame_index, frame_data = item
            try:
                # print("[DEBUG] [COM] - Processing frame index:", frame_index)
                # Ensure frame data is not None before processing
                if frame_data is None:
                    print(f"[PROCESS {processing.current_process().name}] Received None frame data for index {frame_index}. Skipping.")
                    results_queue.put(None)
                    continue

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
                    results_queue.put(None)
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

                results_queue.put(row)
                #print(f"Size of result_queue: {results_queue.qsize()}")

            except Exception as e:
                print(f"Error processing frame {frame_index}: {e}")
                print(traceback.format_exc())
                results_queue.put(None)
                break

def frame_reader(frame_queue: processing.Queue, video_path):
    print(f"[READER] Attempting to open video: '{video_path}'") # <-- ADD THIS LINE
    with open("lag.txt", "r") as f:
        lag = int(f.read().strip())  
        print(f"Got the lag from vector overlay! {lag}")
    cap = cv2.VideoCapture(video_path)
    if lag >= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, lag)  # Skip (lag - 10) frames before video to prevent errors with multiple people in the video!
    # else:
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, lag)  # Start from the lag if lag is less than 10
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    index = 0
    print("[INFO] Loading frames from video...")
    while cap.isOpened():
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
    for _ in range(processing.cpu_count()):
        frame_queue.put(None)
    print("[READER] Sent all sentinels.")

class Processor:
    def __init__(self, video_path):
        #self.cam:cv2.VideoCapture = cam

        self.video_path = video_path
        
        

        
        
    
    def SaveToTxt(self, sex, filename, confidencelevel=0.85, displayCOM=False):
        startTime = time.time()
        frame_queue = Queue()  
        result_queue = Queue()

        reader_thread = threading.Thread(target=frame_reader, args=(frame_queue, self.video_path))
        reader_thread.start()

        # Start worker processes
        workers: list[Process] = []
        num_workers = processing.cpu_count()
        for i in range(num_workers):
            print(f"[PROCESS] Starting process {i}")
            p = processing.Process(target=process_frame, args=(frame_queue, result_queue, sex, confidencelevel, displayCOM))
            p.start()
            workers.append(p)

        reader_thread.join()
        print("[MAIN] Reader thread finished")

        print(f"Size of result_queue: {result_queue.qsize()}")
        # Drain result queue
        output = []
        sentinel_count = 0
        while sentinel_count < num_workers:
            try:
                item = result_queue.get() # Add a timeout here for debugging
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

        output.sort(key=lambda x: x["frame_index"])
        df = pd.DataFrame(output)
        df.to_csv(filename, index=False)

        endTime = time.time()
        print("[INFO] COM Total time taken: ", endTime - startTime)



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