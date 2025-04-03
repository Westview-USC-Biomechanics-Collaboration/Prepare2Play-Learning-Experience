import cv2
import numpy as np
import time

startTime = time.time()

# Load video
cap = cv2.VideoCapture('/home/chaser/Downloads/tss_rl_JG_vid02.mov')

# Initialize variables
frame_index = 0
lowPixel = []

# Define color range for darker green tennis ball (adjust based on lighting conditions)
lower_bound = np.array([25, 130, 50])  # Darker green lower bound in HSV
upper_bound = np.array([80, 240, 140])  # Upper bound in HSV (limit brightness)

# Set area threshold to filter out noise (adjust as needed)
MIN_AREA_THRESHOLD = 500  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for tennis ball color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Filter out small contours based on area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA_THRESHOLD]

        if valid_contours:
            # Find the largest valid contour by area
            largest_contour = max(valid_contours, key=cv2.contourArea)
            
            # Find the lowest point in the contour
            lowest_point = max(largest_contour, key=lambda p: p[0][1])  # p[0][1] is the y-coordinate
            
            # Store frame index and lowest position
            lowPixel.append({"index": frame_index, "position": int(lowest_point[0][1])})

            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Draw rectangle around the ball
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # # Display the frame
    # cv2.imshow('Tennis Ball Tracking', cv2.resize(frame, None, fx=0.5, fy=0.5))
    # cv2.imshow('Mask', cv2.resize(mask, None, fx=0.5, fy=0.5))

    # # Press 'q' to exit
    # if cv2.waitKey(30) & 0xFF == ord('q'):
    #     break
    
    frame_index += 1

# Close all windows
cv2.destroyAllWindows()

# find proccessing time
print(f"Process time: {time.time()-startTime} s")

# Find the dictionary with the highest "position" value
if lowPixel:
    highest_position_entry = max(lowPixel, key=lambda x: x["position"])
    print(highest_position_entry)

    # Display the frame at the lowest ball position
    cap.set(cv2.CAP_PROP_POS_FRAMES, highest_position_entry["index"])
    _, frame = cap.read()

    while True:
        cv2.imshow("Contact Frame", cv2.resize(frame,None, fx=0.5,fy=0.5))
        
        # Press 'q' to exit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    print(f"The ball reaches its lowest position at frame index: {highest_position_entry['index']}")

# Release resources
cap.release()
cv2.destroyAllWindows()
