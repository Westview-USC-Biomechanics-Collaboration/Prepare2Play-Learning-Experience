import cv2
import numpy as np

# Open the video file
# image = cv2.imread('data/derenShoot.png')

posX = []
posY = []

cap = cv2.VideoCapture('data/derenBasketballTest1.mp4')

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# orange: 87, 138, 252
lower_color = np.array([0, 150, 155])
upper_color = np.array([100, 250, 255])

# slow the video down
fps = cap.get(cv2.CAP_PROP_FPS)
slow_factor = 2

delay = int((1000 / fps) * slow_factor)

# #scale size down
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

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            posX.append(cX)
            posY.append(cY)

            cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

    for (x, y) in zip(posX, posY):
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # Apply the mask to the frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # resize the frame
    small_result = cv2.resize(result, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    # Write the frame to the output video file
    out.write(small_result)

    # Display the resulting frame (optional)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
# cv2.waitKey(0)
cv2.destroyAllWindows()
