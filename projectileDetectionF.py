import cv2
import numpy as np
import math

# Open the video file
cap = cv2.VideoCapture('data/derenBasketballTest1.mp4')

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('data/output_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Define the color range for detection
lower_color = np.array([0, 150, 155])
upper_color = np.array([100, 250, 255])

# Contour centroids
posX = []
posY = []

# Contour centroids - average
centroidX = []
centroidY = []
framenumber = []
initialv = 0
illegal = 0

# Get fps
fps = cap.get(cv2.CAP_PROP_FPS)
frame_counter = 0

# Scale size down
scale_factor = 0.5

# Create a named window with the ability to resize
cv2.namedWindow('Resized Video Window', cv2.WINDOW_NORMAL)

# Resize the window
cv2.resizeWindow('Resized Video Window', 980, 540)

# Define a minimum distance to consider contours close to each other
min_distance = 50

# Go over the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_counter += 1

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the defined color range
    mask = cv2.inRange(hsv, lower_color, upper_color)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on proximity to other contours
    filtered_contours = []
    for i, contour in enumerate(contours):
        M1 = cv2.moments(contour)
        if M1["m00"] != 0:
            cX1 = int(M1["m10"] / M1["m00"])
            cY1 = int(M1["m01"] / M1["m00"])
            for j, other_contour in enumerate(contours):
                if i != j:
                    M2 = cv2.moments(other_contour)
                    if M2["m00"] != 0:
                        cX2 = int(M2["m10"] / M2["m00"])
                        cY2 = int(M2["m01"] / M2["m00"])
                        distance = np.sqrt((cX2 - cX1) ** 2 + (cY2 - cY1) ** 2)
                        if distance < min_distance:
                            filtered_contours.append(contour)
                            break

    # Draw out filtered contours and append coordinates into posX and posY
    sumx = 0
    sumy = 0
    legal = 0
    for contour in filtered_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if initialv != 0:
                posX.append(cX)
                posY.append(cY)
            sumx += cX
            sumy += cY
            legal += 1

            if initialv != 0:
                cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)  # Draw on the original frame

    # Begin initial velocity detection
    if legal != 0:
        centroidX.append(sumx / legal)
        centroidY.append(sumy / legal)
        framenumber.append(frame_counter)

    if initialv == 0 and len(centroidX) > 1:
        dx = (centroidX[-1] - centroidX[-2])/395
        dy = (centroidY[-1] - centroidY[-2]) /395
        t = (framenumber[-1] - framenumber[-2])/fps

        vx = dx/t
        vy = dy/t

        #tune this threshold
        if vx**2 + vy**2 >= 10:
            initialv = vx**2 + vy**2 
            print("Initial Velocity: ", initialv)
        else:
            print("Curr Velocity", vx**2 + vy**2)
    for (x, y) in zip(posX, posY):
        if initialv != 0:
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # Apply the mask to the entire frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Resize the frame
    small_result = cv2.resize(result, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    
    # Draw out path of ball
    if initialv != 0 and len(posX) > 2 and len(posY) > 2:
        posX_np = np.array(posX)
        posY_np = np.array(posY)

        coefficients = np.polyfit(posX_np, posY_np, 2)

        a, b, c = coefficients

        x_range = np.linspace(0, 1920, 1000)

        y_values = np.polyval(coefficients, x_range)

        for (x, y) in zip(x_range, y_values):
            if initialv != 0:
                cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)

    # Write the frame to the output video file
    out.write(frame)

    # Display the resulting frame (optional)
    cv2.imshow('Resized Video Window', frame)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# threshold 4.4704 qmeter/second
# approximation: 395 pixels -> 1 meter

# Close all OpenCV windows
cv2.destroyAllWindows()
