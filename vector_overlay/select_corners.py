import cv2

# Globals for cursor position
cursor_x, cursor_y = 0, 0

# Function to zoom in on a region and draw a blue cross
def get_zoomed_region(image, x, y, zoom_size=50, zoom_factor=2):
    h, w = image.shape[:2]
    x1, y1 = max(0, x - zoom_size), max(0, y - zoom_size)
    x2, y2 = min(w, x + zoom_size), min(h, y + zoom_size)

    # Extract the region
    region = image[y1:y2, x1:x2]
    zoomed_region = cv2.resize(region, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

    # Draw a blue cross in the center of the zoomed region
    zh, zw = zoomed_region.shape[:2]
    center_x, center_y = zw // 2, zh // 2
    color = (255, 0, 0)  # Blue color in BGR

    # Horizontal line
    cv2.line(zoomed_region, (center_x - 10, center_y), (center_x + 10, center_y), color, 2)
    # Vertical line
    cv2.line(zoomed_region, (center_x, center_y - 10), (center_x, center_y + 10), color, 2)

    return zoomed_region

def select_points(cap, num_points=8, zoom_size=50, zoom_factor=2):
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the frame.")
        return

    # Resize the frame by a scale of 0.5
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    # List to store the points
    points = []

    # Click event to select points
    def mouse_callback(event, x, y, flags, param):
        global cursor_x, cursor_y
        if event == cv2.EVENT_LBUTTONDOWN:
            # Capture clicked points
            points.append([int(x * 2), int(y * 2)])  # Scale back to original size
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Main Window', frame)

            if len(points) == num_points:
                cv2.destroyWindow('Main Window')
                cv2.destroyWindow('Zoom Window')
        if event == cv2.EVENT_MOUSEMOVE:
            # Update cursor position
            cursor_x, cursor_y = x, y

    # Set mouse callbacks
    cv2.namedWindow('Main Window', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Main Window', mouse_callback)

    cv2.namedWindow('Zoom Window', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Zoom Window', cv2.WND_PROP_TOPMOST, 1)  # Keep on top

    global cursor_x, cursor_y
    cursor_x, cursor_y = 0, 0

    while len(points) < num_points:
        # Show the main window
        cv2.imshow('Main Window', frame)

        # Get the zoomed-in region using the helper function
        zoomed = get_zoomed_region(frame, cursor_x, cursor_y)

        # Display the zoomed-in view
        cv2.imshow('Zoom Window', zoomed)
        cv2.moveWindow('Zoom Window', cursor_x + 200, cursor_y - 100)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save the points to a file
    with open('selected_points.txt', 'w') as f:
        for point in points:
            f.write(f'{point[0]},{point[1]}\n')

    print("Points saved to selected_points.txt")
    return points

if __name__ == "__main__":
    cap = cv2.VideoCapture("/home/chaser/Videos/test_video.mp4")
    select_points(cap, num_points=8)
