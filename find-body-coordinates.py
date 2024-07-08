import cv2
import mediapipe as mp
from openpyxl import Workbook

# Initialize Mediapipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)

# Open the video file.
video_path = 'data/derenBasketballTest1.mp4'
cap = cv2.VideoCapture(video_path)

# Get the frames per second (fps) of the video.
fps = cap.get(cv2.CAP_PROP_FPS)

# Create a new Excel workbook and select the active worksheet.
wb = Workbook()
ws = wb.active
ws.title = "Body Landmarks"

# Define a dictionary to map landmark indices to their names.
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

# Create a window to display the video frame.
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", 800, 600)

headers = ["frame"] + [f"{landmark}_x" for landmark in pose_landmark_names.values()] + [f"{landmark}_y" for landmark in pose_landmark_names.values()]
ws.append(headers)

frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    print(f"Processing frame {frame_number}")

    # Convert frame to RGB for Mediapipe processing.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get pose landmarks.
    results = pose.process(frame_rgb)

    # Extract landmarks and store them in a dictionary.
    landmark_coords = {}
    if results.pose_landmarks:
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            landmark_name = pose_landmark_names.get(id, f"landmark_{id}")
            landmark_coords[f"{landmark_name}_x"] = landmark.x if landmark.visibility > 0 else -1
            landmark_coords[f"{landmark_name}_y"] = landmark.y if landmark.visibility > 0 else -1

            # Draw landmarks on the frame.
            h, w, c = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # Draw a green circle at each landmark position

    # Display the frame with landmarks.
    cv2.imshow("Frame", frame)

    # Write the frame number and landmark coordinates to the Excel file.
    row_data = [frame_number] + [landmark_coords.get(f"{landmark}_x", "") for landmark in
                                 pose_landmark_names.values()] + [landmark_coords.get(f"{landmark}_y", "") for landmark
                                                                  in pose_landmark_names.values()]
    ws.append(row_data)

    # Wait for a key press and check if 'q' is pressed to exit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources.
cap.release()
pose.close()

# Save the workbook.
wb.save("outputs/body_landmarks_from_video.xlsx")

# Close all OpenCV windows.
cv2.destroyAllWindows()
