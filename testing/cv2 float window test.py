import cv2
import numpy as np


# Callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    global cursor_x, cursor_y
    if event == cv2.EVENT_MOUSEMOVE:
        cursor_x, cursor_y = x, y


# Function to zoom in on a region
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



# Load your image
image = cv2.imread(r'C:\Users\16199\Documents\GitHub\Prepare2Play-Learning-Experience-3\lookBack.jpg')
if image is None:
    print("Image not found. Check the file path.")
    exit()

cursor_x, cursor_y = 0, 0

# Create main window and set mouse callback
cv2.namedWindow('Main Window')
cv2.setMouseCallback('Main Window', mouse_callback)
# Create main window
cv2.namedWindow('Main Window', cv2.WINDOW_NORMAL)
cv2.namedWindow('Zoomed View', cv2.WINDOW_NORMAL)

# Keep the Zoomed View window always on top
cv2.setWindowProperty('Zoomed View', cv2.WND_PROP_TOPMOST, 1)
while True:
    # Display the main image
    cv2.imshow('Main Window', image)

    # Get zoomed-in region
    zoomed = get_zoomed_region(image, cursor_x, cursor_y)

    # Display the zoomed-in view as a floating window
    cv2.imshow('Zoomed View', zoomed)
    cv2.moveWindow('Zoomed View', cursor_x + 50, cursor_y - 50)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
