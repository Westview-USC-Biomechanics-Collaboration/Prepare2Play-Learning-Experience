import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture("data/derenBasketballTest1.mp4")

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# orange: 87, 138, 252
lower_color = np.array([0, 110, 155])
upper_color = np.array([10, 250, 255])

# slow the video down
fps = cap.get(cv2.CAP_PROP_FPS)
slow_factor = 2

delay = int((1000 / fps) * slow_factor)

# scale size down
scale_factor = 0.5

# go over the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the defined color range
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Apply the mask to the frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # resize the frame
    small_result = cv2.resize(result, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    # Write the frame to the output video file
    out.write(small_result)

    bw = cv2.cvtColor(small_result, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(small_result, contours, -1, (0, 255, 0), 3)

    # Display the resulting frame (optional)
    cv2.imshow('Frame', small_result)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
