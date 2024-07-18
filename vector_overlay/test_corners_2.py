import cv2
import numpy as np

def display_grey_intensity(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while True:
        # Read the next frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Normalize the intensity values to be displayed
        normalized_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Create a color map to visualize the intensity values
        colored_gray = cv2.applyColorMap(normalized_gray, cv2.COLORMAP_JET)
        
        # Display the original frame and the intensity map
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Grey Intensity', colored_gray)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function with your video path
video_path = 'D:/Users/draar/OneDrive/Documents/GitHub/Prepare2Play-Learning-Experience/data/gis_lr_CC_vid03.mp4'
display_grey_intensity(video_path)
