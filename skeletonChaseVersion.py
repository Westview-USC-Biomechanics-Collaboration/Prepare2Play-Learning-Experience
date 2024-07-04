import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2



def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

# *********************************************************

# img = cv2.imread(filename = 'girl-4051811_960_720.jpg')
# if img is None:
#   print('no image found')
# else:
#   print('success')
#   cv2.imshow('Image Window', img)
#   cv2.waitKey(0)

# *********************************************************
# This part of code is for image only
# # STEP 1: Import the necessary modules.
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
#
# # STEP 2: Create an PoseLandmarker object.
# base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
# options = vision.PoseLandmarkerOptions(
#     base_options=base_options,
#     output_segmentation_masks=True)
# detector = vision.PoseLandmarker.create_from_options(options)
#
# # STEP 3: Load the input image.
# # image = mp.Image.create_from_file("image.jpg")
# image = mp.Image.create_from_file("data/girl-4051811_960_720.jpg")
#
# # STEP 4: Detect pose landmarks from the input image.
# detection_result = detector.detect(image)
#
# # STEP 5: Process the detection result. In this case, visualize it.
# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# cv2.imshow('windowname',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)

# Initialize MediaPipe PoseLandmarker
base_options = python.BaseOptions(model_asset_path='Mediapipe-pose_landmarker/pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# OpenCV video capture
cap = cv2.VideoCapture('data/bcp_lr_CC_vid02.mp4')

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video details
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
output_filename = 'outputs/output_skeleton_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB format expected by MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(mp.ImageFormat.SRGB, rgb_frame)

    # Detect pose landmarks
    detection_result = detector.detect(image)

    # Draw landmarks on the frame
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    # Convert RGB image back to BGR for OpenCV display
    bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # Write the annotated frame to the output video
    out.write(bgr_annotated_image)

    # Display the frame (optional, for visualization)
    cv2.imshow('Skeleton Output', bgr_annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()