import cv2
from PIL import Image

# Function to select points
def select_points(video_path, num_points=8):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Read the first frame

    
    ret, frame = cap.read()
    height, _, _ = frame.shape
    frame = frame[height//2:, :, :]

    
    # Check if frame is read correctly
    if not ret:
        print("Error: Could not read the frame.")
        return
    
    # List to store the points
    points = []
    
    # Function to capture click events
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y + height//2))
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Frame', frame)
            
            if len(points) == num_points:
                cv2.destroyWindow('Frame')
    
    # Display the frame and set the click event handler
    cv2.imshow('Frame', frame)
    cv2.setMouseCallback('Frame', click_event)
    
    # Wait until all points are selected or window is closed
    while len(points) < num_points:
        cv2.waitKey(1)
    
    # Release the video capture
    cap.release()
    
    # Print the selected points
    print("Selected Points: ", points)
    
    # Save the points to a file
    with open('selected_points.txt', 'w') as f:
        for point in points:
            f.write(f'{point[0]},{point[1]}\n')
    
    print("Points saved to selected_points.txt")

# Example usage
# select_points('D:/Users/draar/OneDrive/Documents/GitHub/Prepare2Play-Learning-Experience/data/gis_lr_CC_vid03.mp4')
