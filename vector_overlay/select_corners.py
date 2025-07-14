import numpy as np
import cv2

# Globals for cursor position
cursor_x, cursor_y = 0, 0

# Function to zoom in on a region and draw a blue cross
def get_zoomed_region(frame, x, y, zoom_size=50, zoom_factor=2):
    h, w = frame.shape[:2]
    half_size = zoom_size // 2

    # Coordinates of the ROI in the original frame
    x1 = max(0, x - half_size)
    y1 = max(0, y - half_size)
    x2 = min(w, x + half_size)
    y2 = min(h, y + half_size)

    roi = frame[y1:y2, x1:x2]

    # Create a black image of the zoom_size x zoom_size (original scale)
    black_canvas = np.zeros((zoom_size, zoom_size, 3), dtype=np.uint8)
 
    # Compute the placement coordinates on the black canvas
    roi_h, roi_w = roi.shape[:2]
    y_offset = (zoom_size - roi_h) // 2
    x_offset = (zoom_size - roi_w) // 2

    # Place the ROI in the center of the black canvas
    black_canvas[y_offset:y_offset+roi_h, x_offset:x_offset+roi_w] = roi

    # Resize the canvas to zoom in
    zoomed = cv2.resize(black_canvas, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

    # Draw blue crosshair in the center
    zh, zw = zoomed.shape[:2]
    center_x, center_y = zw // 2, zh // 2
    color = (255, 0, 0)  # Blue in BGR
    thickness = 1

    cv2.line(zoomed, (center_x, 0), (center_x, zh), color, thickness)
    cv2.line(zoomed, (0, center_y), (zw, center_y), color, thickness)

    return zoomed

# def select_points(cap, num_points=8, zoom_size=50, zoom_factor=2):
#     # Check if video opened successfully
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     # Read the first frame
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read the frame.")
#         return

#     # Resize the frame by a scale of 0.5
#     frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

#     # List to store the points
#     points = []

#     # Click event to select points
#     def mouse_callback(event, x, y, flags, param):
#         global cursor_x, cursor_y
#         if event == cv2.EVENT_LBUTTONDOWN:
#             # Capture clicked points
#             points.append([int(x * 2), int(y * 2)])  # Scale back to original size
#             cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
#             cv2.imshow('Main Window', frame)

#             if len(points) == num_points:
#                 cv2.destroyWindow('Main Window')
#                 cv2.destroyWindow('Zoom Window')
#         if event == cv2.EVENT_MOUSEMOVE:
#             # Update cursor position
#             cursor_x, cursor_y = x, y

#     # Set mouse callbacks
#     cv2.namedWindow('Main Window', cv2.WINDOW_NORMAL)
#     cv2.setMouseCallback('Main Window', mouse_callback)

#     cv2.namedWindow('Zoom Window', cv2.WINDOW_NORMAL)
#     cv2.setWindowProperty('Zoom Window', cv2.WND_PROP_TOPMOST, 1)  # Keep on top

#     global cursor_x, cursor_y
#     cursor_x, cursor_y = 0, 0

#     while len(points) < num_points:
#         # Show the main window
#         cv2.imshow('Main Window', frame)

#         # Get the zoomed-in region using the helper function
#         zoomed = get_zoomed_region(frame, cursor_x, cursor_y)

#         # Display the zoomed-in view
#         cv2.imshow('Zoom Window', zoomed)
#         cv2.moveWindow('Zoom Window', cursor_x + 200, cursor_y - 100)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Save the points to a file
#     with open('selected_points.txt', 'w') as f:
#         for point in points:
#             f.write(f'{point[0]},{point[1]}\n')

#     print("Points saved to selected_points.txt")
#     return points

def select_points(cap, num_points=8, zoom_size=50, zoom_factor=2):
    import cv2
    import numpy as np

    # Define yellow color range in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    # Open the video
    cap = cv2.VideoCapture("/home/chaser/Videos/test_video.mp4")
    ret, frame = cap.read()

    if not ret:
        print("Error reading video")
        exit()

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a binary mask where yellow colors are white
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Morphological operations to clean up noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # Find all contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare to save coordinates
    coords = []

    # Minimum area to filter out noise
    min_area = 500  

    #save corners
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            coords.append([x, y])
            coords.append([x + w, y])
            coords.append([x, y + h])
            coords.append([x + w, y + h])
            # Draw rectangle on the original frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save coordinates to a file
    with open("yellow_rectangles_coordinates.txt", "w") as f:
        for x, y in coords:
            f.write(f"{x},{y}\n")

    print(f"{len(coords)} yellow rectangles saved to file.")

    # Show the result
    cv2.imshow("Detected Yellow Rectangles", frame)
    cv2.imshow("Yellow Mask", mask)
    cv2.destroyAllWindows()

    return coords

if __name__ == "__main__":
    cap = cv2.VideoCapture("/home/chaser/Videos/test_video.mp4")
    select_points(cap, num_points=8)
