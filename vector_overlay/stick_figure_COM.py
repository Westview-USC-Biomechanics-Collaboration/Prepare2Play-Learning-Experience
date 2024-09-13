import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from Cal_COM import calculateCOM

"""""
This is the skeleton overlay/stick figure
You need to scroll down to the last line an change the value of video_path in order to process the stick figure
"""""
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
            11: "left_shoulder",
            12: "right_shoulder",
            13: "left_elbow",
            14: "right_elbow",
            15: "left_wrist",
            16: "right_wrist",
            17: "left_pinky",
            18: "right_pinky",
            19: "left_index",
            20: "right_index",
            21: "left_thumb",
            22: "right_thumb",
            23: "left_hip",
            24: "right_hip",
            25: "left_knee",
            26: "right_knee",
            27: "left_ankle",
            28: "right_ankle",
            29: "left_heel",
            30: "right_heel",
            31: "left_foot_index",
            32: "right_foot_index"
        }

def find_coordinates(video_path, sex, filename, confidencelevel=0.85, displayname=False, displaystickfigure=False,
                     displayCOM=False):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=confidencelevel, model_complexity=2)

    cap = cv2.VideoCapture(video_path)
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

    #frame data
    raw_frames = []
    annotated_frames = []

    #position data
    position_data = []
    def update_frame(raw_list):
        # raw list means the raw_frames list
        # Define connections between landmarks
        pose_connections = mp.solutions.pose.POSE_CONNECTIONS
        frame = raw_list[0]
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
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    data.append(cx)
                    data.append(cy)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame to find pose landmarks
        results = pose.process(frame)
        pose_landmarks_list = [results.pose_landmarks] if results.pose_landmarks else []

        # store raw frames
        raw_frames.append(frame)

        # Draw landmarks on the frame
        annotated_frame,datain,dataout = draw_landmarks_on_image(np.copy(frame), pose_landmarks_list, sex, displayname,
                                                  displaystickfigure, displayCOM)
        # store annotated frames, datain, dataout
        annotated_frames.append(annotated_frame)
        datain["COM"] = dataout
        position_data.append(datain)
        print(f"datain in the while loop: {datain}\ndataout in the while loop: {dataout}")
        # Write the annotated frame to the output video file
        # out.write(annotated_frame)

        # Display the annotated frame (optional)
        cv2.imshow('Annotated Frame', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # Press 'Esc' to exit
            break
        elif key == 83:  # Right arrow (next frame)
            pass
        elif key == 81:  # Left arrow (previous frame)
            pass


    # Release video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"frame objects:{raw_frames}")
    print(f"position data: {position_data}")

    current_frame = 0
    while True:
        #check current frame is in range
        if (current_frame>frame_count-1):
            current_frame = frame_count-1
            print(f"This is the last frame!")
        pass
        cv2.imshow("manual correction", annotated_frames[current_frame])
        key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press

        if key == 27:  # Press 'Esc' to exit
            break
        elif key == 83:  # Right arrow (next frame)
            # Increment current_frame, or handle next frame logic
            current_frame += 1
            print(f"current frame: {current_frame + 1}")
        elif key == 81:  # Left arrow (previous frame)
            # Decrement current_frame, or handle previous frame logic
            current_frame -= 1
            print(f"current frame: {current_frame + 1}")







def draw_landmarks_on_image(annotated_image, pose_landmarks_list, sex, displayname=False, displaystickfigure=False,
                            displayCOM=False):
    # Define connections between landmarks
    pose_connections = mp.solutions.pose.POSE_CONNECTIONS
    joints = []
    COM = []
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

        # convert data type
        datain = pd.Series(data, index=columns_name, name="Datain Series")
        dataout = calculateCOM(datain, sex)

        # store data
        joints.append(datain)
        COM.append(dataout)
        # Find COM
        if displayCOM == True:
            cv2.circle(annotated_image, (int(dataout[0]), int(dataout[1])), 12, (0, 0, 255), -1)

    return annotated_image, joints, COM



"""
The function takes in video path, output file name, sex. 
for "sex" parameter, it has to be either "m" or "f"
You can decide to display name of joints, stick figure, or center of mass

use "\\" if you are in windows
use "/" if you are in ios

Set the display element below
"""
# Example usage:
video_path = "C:\\Users\\16199\Desktop\data\Outputs\\Trimmed of fot_Ir_UG_long_vid01_vector_overlay.mp4"  # Replace with your input video file path

displayname = False
displaystickfigure = True
displayCOM = True

filename = "C:\\Users\\16199\Desktop\data\Outputs\\" + video_path.split("\\")[-1][:-4]
# adjust file name
if displaystickfigure == True and displayCOM == True:
    filename += "_COM+Stickfigure.mp4"
elif displayCOM == True and displaystickfigure == False:
    filename += "_COM_only.mp4"
else:
    filename += "_COM.mp4"
find_coordinates(video_path, "m", filename=filename, displayname=displayname, displaystickfigure=displaystickfigure, displayCOM=displayCOM)
print(f"This is output path: {filename}")