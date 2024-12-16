import cv2
import ctypes
# Function to select points
def select_points(cap, num_points=8):

    # Load the user32 library for GetDpiForSystem (Windows 10+)
    user32 = ctypes.windll.user32
    user32.GetDpiForSystem.restype = ctypes.c_uint

    # Get the system DPI
    try:
        dpi = round(user32.GetDpiForSystem()/96.0,1)
    except:
        dpi = 1.5
    print(f"#Assuming 1.0 is 96 dpi\nCurrent system dpi is {dpi}")

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read the first frame
    ret, frame = cap.read()
    height, _, _ = frame.shape

    # resize base on dpi
    frame = cv2.resize(frame,(int(1920/dpi),int(1080/dpi)))

    # Check if frame is read correctly
    if not ret:
        print("Error: Could not read the frame.")
        return

    # List to store the points
    points = []

    # Function to capture click events
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:

            points.append([int(x*dpi),int(y*dpi)])

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

    # Print the selected points
    print("Selected Points: ", points)

    # Save the points to a file
    with open('selected_points.txt', 'w') as f:
        for point in points:
            f.write(f'{point[0]},{point[1]}\n')

    print("Points saved to selected_points.txt")

    return points

if __name__ == "__main__":
    cap = cv2.VideoCapture("C:\\Users\\16199\Desktop\data\spu\Trimmed_Front_nishk 01 right.mp4")
    select_points(cap, 8)
