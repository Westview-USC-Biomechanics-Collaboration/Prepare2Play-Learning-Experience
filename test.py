from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('angle.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Saves the file 
    video_path = 'static/uploaded_video.mp4'
    file.save(video_path)

    # Process video and extract angle data using MediaPipe
    angle_data, times = calculate_angle(video_path)

    # Saves data to CSV file
    save_to_csv(angle_data, times)

    return jsonify({'angle_data': angle_data, 'times': times, 'video_path': video_path})

def calculate_angle(video_path):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    angles = []
    times = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Calculate angle between shoulder and trunk
        if results.pose_landmarks:
            # Using Mediapipe
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

            shoulder_vector = np.array([left_shoulder.x - right_shoulder.x, left_shoulder.y - right_shoulder.y])
            hip_vector = np.array([left_hip.x - right_hip.x, left_hip.y - right_hip.y])

            # Calculate angle in rad then degrees
            angle_rad = np.arccos(np.dot(shoulder_vector, hip_vector) / (np.linalg.norm(shoulder_vector) * np.linalg.norm(hip_vector)))
            angle_deg = np.degrees(angle_rad)
            angles.append(angle_deg)
            times.append(frame_idx / fps)

        frame_idx += 1

    cap.release()

    return angles, times

def save_to_csv(angles, times):
    data = {'Time': times, 'Angle': angles}
    df = pd.DataFrame(data)
    df.to_csv('data/angle_data.csv', index=False)

if __name__ == '__main__':
    app.run(debug=True)
