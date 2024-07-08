from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import plotly.graph_objects as go

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

    # Save uploaded file temporarily
    video_path = 'static/uploaded_video.mp4'
    file.save(video_path)

    # Process video and extract angle data using MediaPipe
    angle_data = calculate_angle(video_path)

    return jsonify({'angle_data': angle_data, 'video_path': video_path})

def calculate_angle(video_path):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    angles = []

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
            # Assuming landmark indices based on Mediapipe Pose
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

            shoulder_vector = np.array([left_shoulder.x - right_shoulder.x, left_shoulder.y - right_shoulder.y])
            hip_vector = np.array([left_hip.x - right_hip.x, left_hip.y - right_hip.y])

            # Calculate angle using dot product and arccos
            angle_rad = np.arccos(np.dot(shoulder_vector, hip_vector) / (np.linalg.norm(shoulder_vector) * np.linalg.norm(hip_vector)))
            angle_deg = np.degrees(angle_rad)
            angles.append(angle_deg)

        # Draw landmarks (optional, for visualization)
        mp_drawing = mp.solutions.drawing_utils
        annotated_image = frame.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show annotated image (optional, for debugging)
        cv2.imshow('Annotated Frame', annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return angles

if __name__ == '__main__':
    app.run(debug=True)
