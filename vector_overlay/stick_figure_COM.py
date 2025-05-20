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

    def pose_landmarks_to_row(self,pose_landmarks_list):
        """
        Convert pose_landmarks_list (a list with either one pose_landmarks object or empty)
        into a dict representing one data row with x, y coords for each landmark.
        If no landmarks detected, returns dict with None for each landmark coordinate.

        Output dict keys: landmark_0_x, landmark_0_y, landmark_0_visibility, ..., landmark_32_x, ...
        """

        # Total number of pose landmarks for MediaPipe Pose is 33
        num_landmarks = 33

        # Prepare empty data row with None defaults
        row = {}
        for i in range(num_landmarks):
            row[f"landmark_{i}_x"] = None
            row[f"landmark_{i}_y"] = None
            row[f"landmark_{i}_visibility"] = None

        if not pose_landmarks_list:
            return row  # no landmarks detected, return all None

        pose_landmarks = pose_landmarks_list[0]  # There is only one in the list

        for i, landmark in enumerate(pose_landmarks.landmark):
            row[f"landmark_{i}_x"] = landmark.x
            row[f"landmark_{i}_y"] = landmark.y
            row[f"landmark_{i}_visibility"] = landmark.visibility

        return row

    def SaveToTxt(cam:cv2.VideoCapture, sex, filename, confidencelevel=0.85):

        starTime = time.time()
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=confidencelevel, model_complexity=2)

        cap = cam
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        output = []
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
            # Process the frame to find pose landmarks
            results = pose.process(frame)
            pose_landmarks_list = [results.pose_landmarks] if results.pose_landmarks else []

            # Convert landmarks to dict row
            row = self.pose_landmarks_to_row(pose_landmarks_list)
            row['frame_index'] = frame_index

            # Optional: Add COM calculation if requested
            if displayCOM and pose_landmarks_list:
                # Prepare data for COM calculation
                pose_landmark_names = {
                    0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
                    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer", 7: "left_ear",
                    8: "right_ear", 9: "mouth_left", 10: "mouth_right", 11: "LSHOULDER",
                    12: "RSHOULDER", 13: "LELBOW", 14: "RELBOW", 15: "LWRIST", 16: "RWRIST",
                    17: "left_pinky", 18: "right_pinky", 19: "left_index", 20: "right_index",
                    21: "left_thumb", 22: "right_thumb", 23: "LHIP", 24: "RHIP", 25: "LKNEE",
                    26: "RKNEE", 27: "LANKLE", 28: "RANKLE", 29: "LHEEL", 30: "RHEEL", 31: "LTOE", 32: "RTOE"
                }

                columns_name = []
                data = []
                for i in range(len(pose_landmark_names)):
                    columns_name.append(str(pose_landmark_names[i]) + "_x")
                    columns_name.append(str(pose_landmark_names[i]) + "_y")
                    landmark = pose_landmarks_list[0].landmark[i]
                    # Use None if not visible
                    if landmark.visibility > 0:
                        data.append(landmark.x)
                        data.append(landmark.y)
                    else:
                        data.append(None)
                        data.append(None)

                datain = pd.Series(data, index=columns_name)
                from Cal_COM import calculateCOM  # Import here or globally if already imported
                com_x, com_y = calculateCOM(datain, sex)
                row['COM_x'] = com_x
                row['COM_y'] = com_y
            else:
                row['COM_x'] = None
                row['COM_y'] = None

            output.append(row)
            frame_index += 1

        cap.release()
        pose.close()
        cv2.destroyAllWindows()

        # Write all data to CSV file
        if len(output) > 0:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = output[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(output)
            print(f"Saved landmark data to {filename}")
        else:
            print("No landmark data to save.")
        
        
        # Calculate and print the time taken
        print(f"Time taken: {time.time() - starTime:.2f} seconds")

if __name__ == "__main__":
    # Example usage:
    video_path = r"C:\Users\chase\Downloads\ajp_lr_JN_long_vid.05.mov"  # Replace with your input video file path
    cam = cv2.VideoCapture(video_path)
    processor = Processor(cam)
    """
    The function takes in video path, output file name, sex. 
    for "sex" parameter, it has to be either "m" or "f"
    You can decide to display name of joints, stick figure, or center of mass

    use "\\" if you are in windows
    use "/" if you are in ios

    Set the display element below
    """

    displayname = False
    displaystickfigure =True
    displayCOM = True

    filename = "outputs\\" + video_path.split("\\")[-1][:-4]
    # adjust file name
    if displaystickfigure == True and displayCOM == True:
        filename += "_COM+Stickfigure.mp4"
    elif displayCOM == True and displaystickfigure == False:
        filename += "_COM_only.mp4"
    else:
        filename += "_COM.mp4"
    print("ready to go")
    #find_coordinates(video_path, "m", filename="coord.txt", displayname=displayname, displaystickfigure=displaystickfigure, displayCOM=displayCOM)
    processor.SaveToTxt(cam, "m", filename="coord.txt", displayname=displayname, displaystickfigure=displaystickfigure, displayCOM=displayCOM)
