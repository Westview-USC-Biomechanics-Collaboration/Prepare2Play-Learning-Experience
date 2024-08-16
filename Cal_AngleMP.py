import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    Points are in the form of (x, y).
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Second point (vertex)
    c = np.array(c)  # Third point
    
    # Calculate the angle
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

# Open the video file
video_path = 'data/Trimmed of cdt_lr_IG_vid02.mp4'

# Extract the base name of the input video file
base_name = os.path.splitext(os.path.basename(video_path))[0]

# Opens video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Only use below lines if cropping is necesary - adjust the decimals to crop
crop_x_start = int(width * 0.1) 
crop_x_end = int(width * 0.75) 
# crop_x_start = 0
# crop_x_end = width
crop_y_start = 0
crop_y_end = height  # Adjusted to crop the full height

# Output video writer
output_directory = 'outputs/JointAngleOuputs'
output_video_path = os.path.join(output_directory, f'Angle(MP)OutputVideo_({base_name}).mp4')
frame_size = (crop_x_end - crop_x_start, crop_y_end - crop_y_start)
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

# Data storage for angles
angle_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Crop the frame to the center area
    cropped_frame = frame[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    
    # Convert the frame to RGB
    image_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect poses
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Draw the pose annotation on the cropped frame
        mp_drawing.draw_landmarks(cropped_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Get coordinates for the right hip, knee, and ankle
        # Get coordinates for the right shoulder, hip, knee, and ankle
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

        # Get coordinates for the left shoulder, hip, knee, and ankle
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

        # Calculate the angles for both knees
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

        # Calculate the angles for both hips
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)

        # Calculate the angles for both ankles
        right_ankle_angle = calculate_angle(right_knee, right_ankle, right_foot_index)
        left_ankle_angle = calculate_angle(left_knee, left_ankle, left_foot_index)

        # Display the angles on the frame
        cv2.putText(cropped_frame, f'RK: {int(right_knee_angle)}', 
                    tuple(np.multiply(right_knee, frame_size).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(cropped_frame, f'LK: {int(left_knee_angle)}', 
                    tuple(np.multiply(left_knee, frame_size).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(cropped_frame, f'RH: {int(right_hip_angle)}', 
                    tuple(np.multiply(right_hip, frame_size).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(cropped_frame, f'LH: {int(left_hip_angle)}', 
                    tuple(np.multiply(left_hip, frame_size).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(cropped_frame, f'RA: {int(right_ankle_angle)}', 
                    tuple(np.multiply(right_ankle, frame_size).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(cropped_frame, f'LA: {int(left_ankle_angle)}', 
                    tuple(np.multiply(left_ankle, frame_size).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Record the angle data with timestamp
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        angle_data.append((timestamp, right_knee_angle, left_knee_angle, right_hip_angle, left_hip_angle, right_ankle_angle, left_ankle_angle))

    # Write the frame to the output video
    out.write(cropped_frame)

cap.release()
out.release()

# Save angle data to a CSV file
# IMPORTANT: HAK - Hip, Ankle, Knee
output_csv_path = os.path.join(output_directory, f'HAKAngleData_({base_name}).csv')
df = pd.DataFrame(angle_data, columns=['Timestamp', 'Right_Knee_Angle', 'Left_Knee_Angle', 'Right_Hip_Angle', 'Left_Hip_Angle', 'Right_Ankle_Angle', 'Left_Ankle_Angle'])
df.to_csv(output_csv_path, index=False)

print("Processing complete! Output video and angle data saved in the data file.")
