import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture('/home/chaser/Downloads/tss_rl_JG_vid02.mov')

# Initialize variables
lowest_y = float('-inf')
lowest_frame = -1
frame_index = 0

# Define color range for darker green tennis ball (adjust based on lighting conditions)
lower_bound = np.array([25, 130, 50])  # Darker green lower bound in HSV   V:(50:140)
upper_bound = np.array([80, 240, 140])  # Upper bound in HSV (limit brightness)


lowPixel = []
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

    # Initiate list to store lowest pixel

    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Find the lowest point in the contour
        lowest_point = max(largest_contour, key=lambda p: p[0][1])  # p[0][1] is the y-coordinate

        # add the point to lowPixel
        lowPixel.append({"index":frame_index,"position":int(lowest_point[0][1])})

        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Draw rectangle around the ball
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Tennis Ball Tracking', cv2.resize(frame, None, fx=0.5, fy=0.5))
    cv2.imshow('Mask', cv2.resize(mask, None, fx=0.5, fy=0.5))

    
    # Press 'q' to exit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
    frame_index += 1

# close all windows
cv2.destroyAllWindows()


# Find the dictionary with the highest "position" value
highest_position_entry = max(lowPixel, key=lambda x: x["position"])
print(highest_position_entry)

# display the frame
cap.set(cv2.CAP_PROP_POS_FRAMES,highest_position_entry["index"])
_,frame = cap.read()

while True:
    cv2.imshow("contact frame", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"The ball reaches its lowest position at frame index: {highest_position_entry['index']}")
