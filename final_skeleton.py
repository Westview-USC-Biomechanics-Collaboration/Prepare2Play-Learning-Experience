import cv2
import pandas as pd
from Cal_COM import calculateCOM

def draw_points(frame, points, frame_height):
    for point in points:
        x, y = int(point[0]), frame_height - int(point[1])  # Flip the y-axis
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle with radius 5
    return frame

# Calculate coordinates
list = calculateCOM("outputs/body_landmarks_from_video.xlsx", "m")
x_coords = list[0]
y_coords = list[1]

if len(x_coords) != len(y_coords):
    raise ValueError("Both Series must have the same length")

# Create the dictionary
coordinates_dict = {str(i): [x, y] for i, (x, y) in enumerate(zip(x_coords, y_coords))}

# Open the video file
cap = cv2.VideoCapture("outputs/output_skeleton_video.mp4")

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the width and height of the frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object to save the video
out = cv2.VideoWriter('outputs/output_skeleton_COM.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Check if the current frame index is in the coordinates_dict
    if str(frame_count) in coordinates_dict:
        # Draw points on the current frame
        frame_with_points = draw_points(frame, [coordinates_dict[str(frame_count)]], frame_height)

        # Write the frame into the output video file
        out.write(frame_with_points)

        # Display the frame (optional)
        cv2.imshow('Frame with Points', frame_with_points)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Write the original frame if no point is to be drawn
        out.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_count += 1

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete. Output saved as 'output_video.mp4'.")
