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
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Update lowest position
        if y + h > lowest_y:
            lowest_y = y + h
            lowest_frame = frame_index
        
        # Draw rectangle around the ball
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Tennis Ball Tracking', frame)
    cv2.imshow('Mask',mask)
    
    # Press 'q' to exit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
    frame_index += 1

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f'The ball reaches its lowest position at frame index: {lowest_frame}')
