import cv2
import ctypes

# Function to select points
def select_points(cap, num_points=8, zoom_size=100, zoom_factor=2):

    # Load the user32 library for GetDpiForSystem (Windows 10+)
    user32 = ctypes.windll.user32
    ctypes.windll.user32.SetProcessDPIAware()
    user32.GetDpiForSystem.restype = ctypes.c_uint

    # Get the system DPI
    try:
        dpi = round(user32.GetDpiForSystem() / 96.0, 2)
    except:
        print("using default scale factor dpi=1.50")
        dpi = 1.5
    print(f"#Assuming 1.00 is 96 dpi\nCurrent system dpi is {dpi}")

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read the first frame
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # Resize based on DPI
    frame = cv2.resize(frame, (int(width / dpi), int(height / dpi)))

    # Check if frame is read correctly
    if not ret:
        print("Error: Could not read the frame.")
        return

    # List to store the points
    points = []

    # Function to capture click events and show zoomed-in window
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([int(x * dpi), int(y * dpi)])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Frame', frame)

            if len(points) == num_points:
                cv2.destroyWindow('Frame')

        if event == cv2.EVENT_MOUSEMOVE:
            # Get the region around the cursor
            x_start = max(0, int(x - zoom_size / 2))
            y_start = max(0, int(y - zoom_size / 2))
            x_end = min(frame.shape[1], int(x + zoom_size / 2))
            y_end = min(frame.shape[0], int(y + zoom_size / 2))

            # Extract the ROI (Region Of Interest)
            roi = frame[y_start:y_end, x_start:x_end]

            # Zoom into the ROI
            zoomed_in = cv2.resize(roi, (zoom_size * zoom_factor, zoom_size * zoom_factor))

            # Display the zoomed-in window next to the cursor
            zoomed_window_name = 'Zoom Window'
            zoomed_x = x + 20
            zoomed_y = y + 20

            # Create a copy of the frame to not interfere with the original window
            zoom_frame = frame.copy()
            cv2.imshow(zoomed_window_name, zoomed_in)
            cv2.moveWindow(zoomed_window_name, zoomed_x, zoomed_y)

    # Display the frame and set the click event handler
    cv2.imshow('Frame', frame)
    cv2.setMouseCallback('Frame', click_event)

    # Wait until all points are selected or window is closed
    while len(points) < num_points:
        cv2.waitKey(1)

    # Print the selected points
    print("Selected Points: ", points)

    # Save the points to a file
    with open('selected_points.txt', 'w') as f:
        for point in points:
            f.write(f'{point[0]},{point[1]}\n')

    print("Points saved to selected_points.txt")

    return points

if __name__ == "__main__":
    #cap = cv2.VideoCapture("C:\\Users\\16199\\Desktop\\data\\spu\\Trimmed_Front_nishk 01 right.mp4")
    cap = cv2.VideoCapture("\\\\dohome2.pusd.dom\\Home2$\\Student2\\1914840\\Chrome Downloads\\longboard_lr_NS_video01.new2.MOV")
    select_points(cap, 8)
