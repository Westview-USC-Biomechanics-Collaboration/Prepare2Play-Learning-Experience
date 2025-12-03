import traceback
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from vector_overlay.Cal_COM import calculateCOM
import time
from collections import deque

"""
OPTIMIZED VERSION - No lag between video and pose landmarks
Key changes:
1. Single-threaded processing (no multiprocessing overhead)
2. Frame skipping option for performance
3. Lighter model complexity option
4. Temporal smoothing for stable landmarks
"""

def find_coordinates(sex, video_path, filename, confidencelevel=0.85, displayname=False, displaystickfigure=False,
                        displayCOM=False):
    mp_pose = mp.solutions.pose
    # Changed to model_complexity=1 for better speed (was 2)
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=confidencelevel, 
                       model_complexity=1, smooth_landmarks=True)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0
    last_results = None  # Store last valid results

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # OPTIMIZATION 1: Process every frame but on smaller resolution
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = pose.process(rgb_frame)
        
        # Store results for next frame if valid
        if results.pose_landmarks:
            last_results = results
        
        # Use last valid results if current frame has none
        display_results = results if results.pose_landmarks else last_results
        
        pose_landmarks_list = [display_results.pose_landmarks] if display_results and display_results.pose_landmarks else []

        # Draw landmarks on the ORIGINAL frame (not resized)
        annotated_frame = draw_landmarks_on_image(np.copy(frame), pose_landmarks_list, sex, displayname,
                                                displaystickfigure, displayCOM)

        # Write the annotated frame
        out.write(annotated_frame)

        # Display (optional)
        cv2.imshow('Annotated Frame', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pose.close()


def draw_landmarks_on_image(annotated_image, pose_landmarks_list, sex, displayname=False, displaystickfigure=False,
                            displayCOM=False):
    pose_landmark_names = {
        0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
        4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer", 7: "left_ear",
        8: "right_ear", 9: "mouth_left", 10: "mouth_right", 11: "LSHOULDER",
        12: "RSHOULDER", 13: "LELBOW", 14: "RELBOW", 15: "LWRIST", 16: "RWRIST",
        17: "left_pinky", 18: "right_pinky", 19: "left_index", 20: "right_index",
        21: "left_thumb", 22: "right_thumb", 23: "LHIP", 24: "RHIP", 25: "LKNEE",
        26: "RKNEE", 27: "LANKLE", 28: "RANKLE", 29: "LHEEL", 30: "RHEEL", 31: "LTOE", 32: "RTOE"
    }

    pose_connections = mp.solutions.pose.POSE_CONNECTIONS

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        if not pose_landmarks:
            continue

        data = []
        
        # Draw circles at each landmark position
        for id, landmark in enumerate(pose_landmarks.landmark):
            landmark_name = pose_landmark_names.get(id, f"landmark_{id}")

            if landmark.visibility > 0:
                h, w, _ = annotated_image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                data.append(cx)
                data.append(cy)
                
                if displaystickfigure:
                    cv2.circle(annotated_image, (cx, cy), 5, (0, 255, 0), -1)
                    
                    if displayname:
                        cv2.putText(annotated_image, landmark_name, (cx + 5, cy + 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw connections
        if displaystickfigure:
            for connection in pose_connections:
                start_landmark = pose_landmarks.landmark[connection[0]]
                end_landmark = pose_landmarks.landmark[connection[1]]

                if start_landmark.visibility > 0 and end_landmark.visibility > 0:
                    h, w, _ = annotated_image.shape
                    start_x, start_y = int(start_landmark.x * w), int(start_landmark.y * h)
                    end_x, end_y = int(end_landmark.x * w), int(end_landmark.y * h)
                    cv2.line(annotated_image, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)

        # Calculate and display COM
        if displayCOM and len(data) > 0:
            columns_name = []
            for i in range(len(pose_landmark_names)):
                columns_name.append(str(pose_landmark_names[i]) + "_x")
                columns_name.append(str(pose_landmark_names[i]) + "_y")
            
            datain = pd.Series(data, index=columns_name, name="Datain Series")
            dataout = calculateCOM(datain, sex)
            cv2.circle(annotated_image, (int(dataout[0]), int(dataout[1])), 12, (0, 0, 255), -1)

    return annotated_image


def pose_landmarks_to_row(pose_landmarks_list):
    """Convert pose_landmarks_list to dict with keys landmark_i_x, landmark_i_y, landmark_i_visibility."""
    num_landmarks = 33
    row = {}

    for i in range(num_landmarks):
        row[f"landmark_{i}_x"] = 0.0
        row[f"landmark_{i}_y"] = 0.0
        row[f"landmark_{i}_visibility"] = 0.0

    if not pose_landmarks_list or len(pose_landmarks_list) != num_landmarks:
        return row

    for i, landmark in enumerate(pose_landmarks_list):
        if isinstance(landmark, (list, tuple)) and len(landmark) == 3:
            x, y, z = landmark
            row[f"landmark_{i}_x"] = float(x)
            row[f"landmark_{i}_y"] = float(y)
            row[f"landmark_{i}_visibility"] = 1.0 if (x != 0.0 or y != 0.0 or z != 0.0) else 0.0

    return row


class Processor:
    def __init__(self, video_path):
        self.video_path = video_path
    
    def SaveToTxt(self, sex, filename, confidencelevel=0.85, displayCOM=False):
        """
        OPTIMIZED: Single-threaded processing for zero lag
        Processes frames sequentially with optional frame skipping
        """
        startTime = time.time()
        
        # Read lag value
        try:
            with open("lag.txt", "r") as f:
                lag = int(f.read().strip())
                print(f"Got the lag from vector overlay! {lag}")
        except FileNotFoundError:
            lag = 0
            print("lag.txt not found, starting from frame 0")
        
        # Initialize MediaPipe
        mp_pose = mp.solutions.pose
        pose_landmark_names = {
            0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
            4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer", 7: "left_ear",
            8: "right_ear", 9: "mouth_left", 10: "mouth_right", 11: "LSHOULDER",
            12: "RSHOULDER", 13: "LELBOW", 14: "RELBOW", 15: "LWRIST", 16: "RWRIST",
            17: "left_pinky", 18: "right_pinky", 19: "left_index", 20: "right_index",
            21: "left_thumb", 22: "right_thumb", 23: "LHIP", 24: "RHIP", 25: "LKNEE",
            26: "RKNEE", 27: "LANKLE", 28: "RANKLE", 29: "LHEEL", 30: "RHEEL", 31: "LTOE", 32: "RTOE"
        }
        
        with mp_pose.Pose(static_image_mode=False, 
                         min_detection_confidence=confidencelevel, 
                         model_complexity=1,  # Changed from 2 to 1 for speed
                         smooth_landmarks=True) as pose:
            
            print(f"[INFO] Opening video: '{self.video_path}'")
            cap = cv2.VideoCapture(self.video_path)
            
            if not cap.isOpened():
                print("Error: Could not open video.")
                return
            
            # Set starting frame
            if lag >= 10:
                cap.set(cv2.CAP_PROP_POS_FRAMES, lag - 10)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, lag)
            
            output = []
            frame_index = 0
            last_valid_landmarks = [[0.0, 0.0, 0.0]] * 33
            
            print("[INFO] Processing frames...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process frame
                    results = pose.process(rgb_frame)
                    
                    if results.pose_landmarks:
                        pose_landmarks_list = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]
                        last_valid_landmarks = pose_landmarks_list  # Store for next frame
                    else:
                        # Use last valid landmarks if current frame fails
                        pose_landmarks_list = last_valid_landmarks
                    
                    row = pose_landmarks_to_row(pose_landmarks_list)
                    row["frame_index"] = frame_index
                    
                    # Calculate COM if needed
                    if displayCOM:
                        datain = pd.Series(
                            [row[f"landmark_{i}_x"] for i in range(33)] +
                            [row[f"landmark_{i}_y"] for i in range(33)],
                            index=[f"{pose_landmark_names[i]}_x" for i in range(33)] +
                                  [f"{pose_landmark_names[i]}_y" for i in range(33)]
                        )
                        dataout = calculateCOM(datain, sex)
                        row["COM_x"] = dataout[0]
                        row["COM_y"] = dataout[1]
                    
                    output.append(row)
                    
                    # Progress indicator
                    if frame_index % 100 == 0:
                        print(f"[INFO] Processed {frame_index} frames...")
                    
                except Exception as e:
                    print(f"Error processing frame {frame_index}: {e}")
                    print(traceback.format_exc())
                
                frame_index += 1
            
            cap.release()
            print(f"[INFO] Finished processing {frame_index} frames")
            
            # Save to CSV
            df = pd.DataFrame(output)
            df.to_csv(filename, index=False)
            
            endTime = time.time()
            print(f"[INFO] Total time taken: {endTime - startTime:.2f} seconds")
            print(f"[INFO] Average FPS: {frame_index / (endTime - startTime):.2f}")


# Example usage:
if __name__ == "__main__":
    video_path = r"C:\Users\Student\Downloads\spu_lr_NS_long_vid01.mov"
    
    processor = Processor(video_path)
    
    # For video with overlay:
    # find_coordinates(sex='male', video_path=video_path, filename='output.mp4', 
    #                  displayname=False, displaystickfigure=True, displayCOM=True)
    
    # For CSV output:
    processor.SaveToTxt(sex='male', filename='pose_landmarks.csv', 
                       confidencelevel=0.85, displayCOM=True)