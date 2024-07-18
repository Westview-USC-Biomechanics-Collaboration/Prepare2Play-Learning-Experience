import cv2
import numpy as np

def track_box_corners(video_path, initial_corners):
    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the video")
        return

    # Convert to grayscale
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Convert initial_corners to the format expected by OpenCV
    corners = np.array(initial_corners, dtype=np.float32).reshape(-1, 1, 2)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(first_frame)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        new_corners, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, corners, None)

        # Select good points
        good_new = new_corners[status == 1]
        good_old = corners[status == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
        
        img = cv2.add(frame, mask)

        # Print corner coordinates
        print(f"Frame {frame_count} corner coordinates:")
        for i, corner in enumerate(good_new):
            x, y = corner.ravel()
            print(f"Corner {i+1}: ({x:.2f}, {y:.2f})")
        print()

        cv2.imshow('Box Tracking', img)

        # Update the previous frame and previous points
        prev_gray = frame_gray.copy()
        corners = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Usage
video_path = 'D:/Users/draar/OneDrive/Documents/GitHub/Prepare2Play-Learning-Experience/data/gis_lr_CC_vid03.mp4'
initial_corners = [
    (100, 100), (200, 100), (300, 100), (400, 100),  # top 4 corners
    (100, 200), (200, 200), (300, 200), (400, 200)   # bottom 4 corners
]
track_box_corners(video_path, select_points(video_path))