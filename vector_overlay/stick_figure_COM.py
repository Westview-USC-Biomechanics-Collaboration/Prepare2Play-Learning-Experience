import traceback
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from vector_overlay.Cal_COM import calculateCOM
import csv
import time

"""""
This is the skeleton overlay/stick figure
You need to scroll down to the last line an change the value of video_path in order to process the stick figure
"""""

class Processor:
    def __init__(self, cam:cv2.VideoCapture):
        self.cam:cv2.VideoCapture = cam
    

    def find_coordinates(self, sex, filename, confidencelevel=0.85, displayname=False, displaystickfigure=False,
                        displayCOM=False):
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=confidencelevel, model_complexity=2)

        cap = self.cam
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
            annotated_frame = self.draw_landmarks_on_image(np.copy(frame), pose_landmarks_list, sex, displayname,
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


    def draw_landmarks_on_image(self, annotated_image, pose_landmarks_list, sex, displayname=False, displaystickfigure=False,
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


    def pose_landmarks_to_row(self, pose_landmarks_list):
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


    def SaveToTxt(self, sex, filename, confidencelevel=0.85, displayCOM=False):
        startTime = time.time()

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=confidencelevel, model_complexity=2)

        cap = self.cam
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        output = []
        frame_index = 0

        pose_landmark_names = {
            0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
            4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer", 7: "left_ear",
            8: "right_ear", 9: "mouth_left", 10: "mouth_right", 11: "LSHOULDER",
            12: "RSHOULDER", 13: "LELBOW", 14: "RELBOW", 15: "LWRIST", 16: "RWRIST",
            17: "left_pinky", 18: "right_pinky", 19: "left_index", 20: "right_index",
            21: "left_thumb", 22: "right_thumb", 23: "LHIP", 24: "RHIP", 25: "LKNEE",
            26: "RKNEE", 27: "LANKLE", 28: "RANKLE", 29: "LHEEL", 30: "RHEEL", 31: "LTOE", 32: "RTOE"
        }

        while cap.isOpened():
            try:
                print("[DEBUG] [COM] - Processing frame index:", frame_index)
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
                results = pose.process(frame)

                if results.pose_landmarks:
                    # Convert landmarks to list of [x, y, z]
                    pose_landmarks_list = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]
                else:
                    # No pose detected - 33 landmarks all zeros
                    pose_landmarks_list = [[0.0, 0.0, 0.0]] * 33

                row = self.pose_landmarks_to_row(pose_landmarks_list)

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

                output.append(row)

                frame_index += 1
            except Exception as e:
                print(f"Error processing frame {frame_index}: {e}")
                print(traceback.format_exc())
                break


        # Create DataFrame and save to CSV file
        df = pd.DataFrame(output)
        df.to_csv(filename, index=False)

        endTime = time.time()
        print("Total time taken: ", endTime - startTime)


if __name__ == "__main__":
    video_path = r"C:\Users\chase\Downloads\ajp_lr_JN_long_vid.05.mov"  # Change this to your video file path
    cap = cv2.VideoCapture(video_path)
    processor = Processor(cap)

    # Example usage:
    # Draw skeleton with display options:
    # processor.find_coordinates(sex='male', filename='output.mp4', displayname=True, displaystickfigure=True, displayCOM=True)

    # Save landmarks to CSV with zeros for no detection
    processor.SaveToTxt(sex='male', filename='pose_landmarks.csv', confidencelevel=0.85, displayCOM=True)
