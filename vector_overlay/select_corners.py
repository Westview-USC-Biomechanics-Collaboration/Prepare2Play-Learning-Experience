"""
Key changes to select_corners.py to reduce duplicate displays:
1. Store masks in dictionary for single display at end
2. Remove intermediate cv2.imshow() calls
3. Add debug flag for optional visualization
"""

def select_points(self, cap, view, debug=True):
    """
    Modified version that only shows final result unless debug=True
    
    Args:
        cap: Video capture object
        view: String indicating view type ("Long View", "Short View", "Top View")
        debug: If True, show intermediate processing steps
    """
    import cv2
    import numpy as np

    # Define yellow color range in HSV
    lower_yellow = np.array([18, 80, 50])
    upper_yellow = np.array([38, 255, 255])

    ret, frame = cap.read()

    if not cap.isOpened():
        print("Could not open video file.")
        exit()

    if not ret:
        print("Error reading video during selecting corners")
        exit()

    if frame is None:
        print("Error with rect detection")
        exit()

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Store masks for optional display
    masks = {'original': mask.copy()}

    # Morphological operations
    kernel_h = np.ones((1, 100), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h)
    kernel_v = np.ones((1, 1), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_v)
    
    masks['morphed'] = mask.copy()

    # Define ROI based on view
    h, w = mask.shape[:2]
    offset_x, offset_y = 0, 0
    
    if view == "Long View":
        y1, y2 = int(0.7 * h), int(0.9 * h)
        x1, x2 = int(0.25 * w), int(0.75 * w)
    elif view == "Top View":
        y1, y2 = int(0.3 * h), int(0.7 * h)
        x1, x2 = int(0.2 * w), int(0.8 * w)
    else:  # Short View
        y1, y2 = int(0.6 * h), int(0.8 * h)
        x1, x2 = int(0.45 * w), int(0.8 * w)
    
    offset_x, offset_y = x1, y1
    roi = mask[y1:y2, x1:x2]
    masks['roi'] = roi.copy()

    # Only show debug masks if requested
    if debug:
        for name, img in masks.items():
            cv2.namedWindow(f"Debug: {name}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"Debug: {name}", 800, 600)
            cv2.imshow(f"Debug: {name}", img)

    # Find contours in ROI
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coords = []
    min_area = 500 if view == "Short View" else 2000

    # Process contours
    for contour in contours:
        contour += [offset_x, offset_y]
        hull = cv2.convexHull(contour)
        area = cv2.contourArea(hull)

        epsilon = 0.03 if view == "Short View" else 0.01
        epsilon *= cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if area > min_area:
            corners = approx.reshape(-1, 2)
            if len(corners) == 4:
                cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)
                for corner in corners:
                    x, y = corner
                    coords.append([x, y])

    # Sort and arrange corners
    coords_one = sorted(coords, key=lambda x: x[0])[0:2]
    coords_two = sorted(coords, key=lambda x: x[0])[2:]
    coords_one = sorted(coords_one, key=lambda x: x[1])
    coords_two = sorted(coords_two, key=lambda x: x[1])
    coords = coords_one + coords_two

    print(f"Detected {len(coords)} corners for {view}.")
    
    if len(coords) < 4:
        print("Error: Not enough corners detected.")
        return None

    # Calculate middle points between plates
    if view == "Short View":
        coords.append([(coords[2][0] + coords[3][0])/2, (coords[2][1] + coords[3][1])/2 - 25])
        coords.append([(coords[2][0] + coords[3][0])/2, (coords[2][1] + coords[3][1])/2])
        coords.append([(coords[0][0] + coords[1][0])/2, (coords[0][1] + coords[1][1])/2 - 25])
        coords.append([(coords[0][0] + coords[1][0])/2, (coords[0][1] + coords[1][1])/2])
    else:
        coords.append([(coords[0][0] + coords[2][0])/2 - 10, (coords[0][1] + coords[2][1])/2])
        coords.append([(coords[0][0] + coords[2][0])/2 + 10, (coords[0][1] + coords[2][1])/2])
        coords.append([(coords[1][0] + coords[3][0])/2 - 10, (coords[1][1] + coords[3][1])/2])
        coords.append([(coords[1][0] + coords[3][0])/2 + 10, (coords[1][1] + coords[3][1])/2])

    # Rearrange to final order
    output = [[0,0]] * 8
    if view == "Short View":
        output[0], output[1], output[2], output[3] = coords[0], coords[2], coords[4], coords[6]
        output[4], output[5], output[6], output[7] = coords[7], coords[5], coords[3], coords[1]
    else:
        output = [coords[1], coords[6], coords[4], coords[0], 
                  coords[7], coords[3], coords[2], coords[5]]

    # Draw final corners on frame
    for out in output:
        cv2.circle(frame, (int(out[0]), int(out[1])), 5, (0, 0, 255), -1)

    # Save to file
    with open("selected_points.txt", "w") as f:
        for x, y in output:
            f.write(f"{x},{y}\n")

    print(f"Saved {len(output)} coordinates to selected_points.txt")

    # Show final result
    cv2.namedWindow("Detected Yellow Rectangles", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detected Yellow Rectangles", 800, 600)
    cv2.imshow("Detected Yellow Rectangles", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return output

###################################### OLD ###############################################
# import numpy as np
# import cv2

# # Globals for cursor position
# cursor_x, cursor_y = 0, 0

# # Function to zoom in on a region and draw a blue cross
# def get_zoomed_region(frame, x, y, zoom_size=50, zoom_factor=2):
#     h, w = frame.shape[:2]
#     half_size = zoom_size // 2

#     # Coordinates of the ROI in the original frame
#     x1 = max(0, x - half_size)
#     y1 = max(0, y - half_size)
#     x2 = min(w, x + half_size)
#     y2 = min(h, y + half_size)

#     roi = frame[y1:y2, x1:x2]

#     # Create a black image of the zoom_size x zoom_size (original scale)
#     black_canvas = np.zeros((zoom_size, zoom_size, 3), dtype=np.uint8)

#     # Compute the placement coordinates on the black canvas
#     roi_h, roi_w = roi.shape[:2]
#     y_offset = (zoom_size - roi_h) // 2
#     x_offset = (zoom_size - roi_w) // 2

#     # Place the ROI in the center of the black canvas
#     black_canvas[y_offset:y_offset+roi_h, x_offset:x_offset+roi_w] = roi

#     # Resize the canvas to zoom in
#     zoomed = cv2.resize(black_canvas, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

#     # Draw blue crosshair in the center
#     zh, zw = zoomed.shape[:2]
#     center_x, center_y = zw // 2, zh // 2
#     color = (255, 0, 0)  # Blue in BGR
#     thickness = 1

#     cv2.line(zoomed, (center_x, 0), (center_x, zh), color, thickness)
#     cv2.line(zoomed, (0, center_y), (zw, center_y), color, thickness)

#     return zoomed

# # def select_points(cap, num_points=8, zoom_size=50, zoom_factor=2):
# #     # Check if video opened successfully
# #     if not cap.isOpened():
# #         print("Error: Could not open video.")
# #         return
# #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# #     # Read the first frame
# #     ret, frame = cap.read()
# #     if not ret:
# #         print("Error: Could not read the frame.")
# #         return

# #     # Resize the frame by a scale of 0.5
# #     frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

# #     # List to store the points
# #     points = []

# #     # Click event to select points
# #     def mouse_callback(event, x, y, flags, param):
# #         global cursor_x, cursor_y
# #         if event == cv2.EVENT_LBUTTONDOWN:
# #             # Capture clicked points
# #             points.append([int(x * 2), int(y * 2)])  # Scale back to original size
# #             cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
# #             cv2.imshow('Main Window', frame)

# #             if len(points) == num_points:
# #                 cv2.destroyWindow('Main Window')
# #                 cv2.destroyWindow('Zoom Window')
# #         if event == cv2.EVENT_MOUSEMOVE:
# #             # Update cursor position
# #             cursor_x, cursor_y = x, y

# #     # Set mouse callbacks
# #     cv2.namedWindow('Main Window', cv2.WINDOW_NORMAL)
# #     cv2.setMouseCallback('Main Window', mouse_callback)

# #     cv2.namedWindow('Zoom Window', cv2.WINDOW_NORMAL)
# #     cv2.setWindowProperty('Zoom Window', cv2.WND_PROP_TOPMOST, 1)  # Keep on top

# #     global cursor_x, cursor_y
# #     cursor_x, cursor_y = 0, 0

# #     while len(points) < num_points:
# #         # Show the main window
# #         cv2.imshow('Main Window', frame)

# #         # Get the zoomed-in region using the helper function
# #         zoomed = get_zoomed_region(frame, cursor_x, cursor_y)

# #         # Display the zoomed-in view
# #         cv2.imshow('Zoom Window', zoomed)
# #         cv2.moveWindow('Zoom Window', cursor_x + 200, cursor_y - 100)

# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break

# #     # Save the points to a file
# #     with open('selected_points.txt', 'w') as f:
# #         for point in points:
# #             f.write(f'{point[0]},{point[1]}\n')

# #     print("Points saved to selected_points.txt")
# #     return points

# def two_rect_detection(cap, num_points=8, zoom_size=50, zoom_factor=2):
#     import cv2
#     import numpy as np

#     def find_best_rect(contours, min_area=2000):
#         best_rect = None
#         best_score = -1  # higher is better

#         for contour in contours:
#             hull = cv2.convexHull(contour)
#             area = cv2.contourArea(hull)

#             # Approximate contour to polygon to reduce points
#             epsilon = 0.01 * cv2.arcLength(hull, True)  # adjust for precision
#             approx = cv2.approxPolyDP(hull, epsilon, True)

#             if area > min_area:

#                 corners = approx.reshape(-1, 2)  # shape (num_points, 2)

#                 if len(corners) == 4:
#                     total_score = cv2.contourArea(approx)
#             if total_score > best_score:
#                 best_score = total_score
#                 best_rect = contour

#         return best_rect

#     # Define yellow color range in HSV
#     lower_yellow = np.array([20, 100, 100])
#     upper_yellow = np.array([35, 255, 255])

#     # Lower range for red
#     lower_red1 = np.array([0, 120, 70])    
#     upper_red1 = np.array([10, 255, 255])

#     # Upper range for red
#     lower_red2 = np.array([170, 120, 70])  
#     upper_red2 = np.array([180, 255, 255])

#     # Open the video
#     ret, frame = cap.read()
    
#     if not cap.isOpened():
#         print("❌ Could not open video file.")
#         exit()

#     if not ret:
#         print("Error reading video during selecting corners")
#         exit()

#     if frame is None:
#         print("Error with rect detection")
#         exit()

#     # Convert to HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Create a binary mask where yellow colors are white
#     yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

#     mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
#     mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
#     red_mask = cv2.bitwise_or(mask1, mask2)

#     cv2.namedWindow("yellow_original", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("yellow_original", 800, 600)  # Set window size
#     cv2.imshow("yellow_original", yellow_mask)

#     cv2.namedWindow("red_original", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("red_original", 800, 600)  # Set window size
#     cv2.imshow("red_original", red_mask)

#     kernel = np.ones((2,2), np.uint8)
#     yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
#     red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

#     cv2.namedWindow("yellow_opened", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("yellow_opened", 800, 600)  # Set window size
#     cv2.imshow("yellow_opened", yellow_mask)

#     cv2.namedWindow("red_opened", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("red_opened", 800, 600)  # Set window size
#     cv2.imshow("red_opened", red_mask)

#     # Optional: closing to seal any final small gaps
#     kernel_h = np.ones((1, 100), np.uint8)
#     yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel_h)
#     kernel_v = np.ones((1, 1), np.uint8)
#     yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel_v)

#     red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_h)
#     red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_v)  

#     cv2.namedWindow("yellow_edited", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("yellow_edited", 800, 600)  # Set window size
#     cv2.imshow("yellow_edited", yellow_mask)

#     cv2.namedWindow("red_edited", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("red_edited", 800, 600)  # Set window size
#     cv2.imshow("red_edited", red_mask)

#     # Trying to auto create roi
#     yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # # Optional: filter by area
#     y_contours = [c for c in yellow_contours if cv2.contourArea(c) > 2000]
#     r_contours = [c for c in red_contours if cv2.contourArea(c) > 2000]

#     # Prepare to save coordinates
#     yellow_coords =[]
#     red_coords = []

#     # # Get the bounding box of the largest contour
#     if y_contours:
#         c = find_best_rect(y_contours)
#         x, y, w, h = cv2.boundingRect(c)
#         cx, cy = x + w // 2, y + h // 2  # Center of the original box

#         # Scale factors
#         scale_x = 1.2  
#         scale_y = 1.2  

#         # New width and height
#         new_w = int(w * scale_x)
#         new_h = int(h * scale_y)

#         # New top-left corner
#         x1 = max(0, cx - new_w // 2)
#         y1 = max(0, cy - new_h // 2)

#         # New bottom-right corner
#         x2 = min(yellow_mask.shape[1], cx + new_w // 2)
#         y2 = min(yellow_mask.shape[0], cy + new_h // 2)

#         # Extract scaled ROI
#         yellow_roi = yellow_mask[y1:y2, x1:x2]
#         yellow_offset_x, yellow_offset_y = x1, y1  # for mapping back later
#     if r_contours:
#         c = find_best_rect(r_contours)
#         x, y, w, h = cv2.boundingRect(c)
#         cx, cy = x + w // 2, y + h // 2  # Center of the original box

#         # Scale factors
#         scale_x = 1.2  
#         scale_y = 1.2  

#         # New width and height
#         new_w = int(w * scale_x)
#         new_h = int(h * scale_y)

#         # New top-left corner
#         x1 = max(0, cx - new_w // 2)
#         y1 = max(0, cy - new_h // 2)

#         # New bottom-right corner
#         x2 = min(red_mask.shape[1], cx + new_w // 2)
#         y2 = min(red_mask.shape[0], cy + new_h // 2)

#         # Extract scaled ROI
#         red_roi = red_mask[y1:y2, x1:x2]
#         red_offset_x, red_offset_y = x1, y1  # for mapping back later

#     cv2.namedWindow("y_kernel observation", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("y_kernel observation", 800, 600)  # Set window size
#     cv2.imshow("y_kernel observation", yellow_roi)

#     cv2.namedWindow("r_kernel observation", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("r_kernel observation", 800, 600)  # Set window size
#     cv2.imshow("r_kernel observation", red_roi)

#     # # Find all contours in the roi
#     y_contours, _ = cv2.findContours(yellow_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     r_contours, _ = cv2.findContours(red_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     #save corners
#     for contour in y_contours:
#         contour += [yellow_offset_x, yellow_offset_y]
#         hull = cv2.convexHull(contour)

#         # Approximate contour to polygon to reduce points
#         epsilon = 0.01 * cv2.arcLength(hull, True)  # adjust for precision
#         approx = cv2.approxPolyDP(hull, epsilon, True)

#         corners = approx.reshape(-1, 2)  # shape (num_points, 2)

#         if len(corners) == 4:
#             cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)  # Blue polygon outline
#             for corner in corners:
#                 x, y = corner
#                 yellow_coords.append([x, y])

#     for contour in r_contours:
#         contour += [red_offset_x, red_offset_y]
#         hull = cv2.convexHull(contour)

#         # Approximate contour to polygon to reduce points
#         epsilon = 0.03 * cv2.arcLength(hull, True)  # adjust for precision
#         approx = cv2.approxPolyDP(hull, epsilon, True)

#         corners = approx.reshape(-1, 2)  # shape (num_points, 2)

#         if len(corners) == 4:
#             cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2) # Green polygon outline
#             for corner in corners:
#                 x, y = corner
#                 red_coords.append([x, y])
    
    
#     coords_one = sorted(yellow_coords, key=lambda x: (x[1], x[0]))
#     coords_two = sorted(red_coords, key=lambda x: (x[1], x[0]))
#     coords = coords_one + coords_two
#     print(coords)

#     output = [[0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]
#     #rearrange list
#     output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7] = coords[0], coords[1], coords[3], coords[2], coords[4], coords[5], coords[7], coords[6] 

#     print("Plate 1 corners:", output[0:4])
#     print("Plate 2 corners:", output[4:8])

#     for out in yellow_coords:
#         cv2.circle(frame, (int(out[0]), int(out[1])), 5, (0, 0, 255), -1)  # Red dots for corners
#     for out in red_coords:
#         cv2.circle(frame, (int(out[0]), int(out[1])), 5, (255, 255, 255), -1)  # White dots for corners

#     # Save coordinates to a file
#     with open("selected_points.txt", "w") as f:
#         for x, y in output:
#             f.write(f"{x},{y}\n")

#     print(f"{len(output)} coordinates saved to file.")

#     # Show the result
#     cv2.namedWindow("Detected Rectangles", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Detected Rectangles", 800, 600)  # Set window size
#     cv2.imshow("Detected Rectangles", frame)

#     cv2.waitKey(0)  # Wait indefinitely until a key is pressed
#     cv2.destroyAllWindows()

#     return output

# def select_points(self, cap, view):
#     import cv2
#     import numpy as np

#     # Define yellow color range in HSV
#     lower_yellow = np.array([18, 80, 50])
#     upper_yellow = np.array([38, 255, 255])

#     # lower_yellow = np.array([15, 60, 40])
#     # upper_yellow = np.array([45, 255, 255])

#     # Open the video
#     ret, frame = cap.read()

#     # Load an image instead of a video
#     # frame = cv2.imread("vector_overlay\IMG_2518.jpg")

#     if not cap.isOpened():
#         print("Could not open video file.")
#         exit()

#     if not ret:
#         print("Error reading video during selecting corners")
#         exit()

#     if frame is None:
#         print("Error with rect detection")
#         exit()

#     # Convert to HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Create a binary mask where yellow colors are white
#     mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

#     cv2.namedWindow("original", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("original", 800, 600)  # Set window size
#     cv2.imshow("original", mask)

#     # Optional: closing to seal any final small gaps
#     # Horizontal kernel to connect horizontal lines
#     kernel_h = np.ones((1, 100), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h)
#     kernel_v = np.ones((1, 1), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_v)

#     cv2.namedWindow("one", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("one", 800, 600)  # Set window size
#     cv2.imshow("one", mask)

#     # ROI crop
#     # h, w = mask.shape
#     # roi = mask[int(h * 0.40):int(h * 0.85), int(w * 0.01):int(w * 0.85)]
#     # offset_x, offset_y = int(w * 0.01), int(h * 0.4)

#     # # Trying to auto create roi
#     # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # # Optional: filter by area
#     # contours = [c for c in contours if cv2.contourArea(c) > 2000]

#     # # Get the bounding box of the largest contour
#     # if contours:
#     #     c = max(contours, key=cv2.contourArea)
#     #     x, y, w, h = cv2.boundingRect(c)
#     #     cx, cy = x + w // 2, y + h // 2  # Center of the original box

#     #     # Scale factors
#     #     scale_x = 1
#     #     scale_y = 1

#     #     # New width and height
#     #     new_w = int(w * scale_x)
#     #     new_h = int(h * scale_y)

#     #     # New top-left corner
#     #     x1 = max(0, cx - new_w // 2)
#     #     y1 = max(0, cy - new_h // 2)

#     #     # New bottom-right corner
#     #     x2 = min(mask.shape[1], cx + new_w // 2)
#     #     y2 = min(mask.shape[0], cy + new_h // 2)

#     #     # Extract scaled ROI
#     #     roi = mask[y1:y2, x1:x2]
#     #     offset_x, offset_y = x1, y1  # for mapping back later

#     h, w = mask.shape[:2]
#     print(f"The view of the video is {view}")
#     if view == "Long View":
#         y1, y2 = int(0.7 * h), int(0.9 * h)
#         x1, x2 = int(0.25 * w), int(0.75 * w)
#         offset_x, offset_y = x1, y1
#         roi = mask[y1:y2, x1:x2]
#     elif view == "Top View":
#         y1, y2 = int(0.3 * h), int(0.7 * h)
#         x1, x2 = int(0.2 * w), int(0.8 * w)
#         offset_x, offset_y = x1, y1
#         roi = mask[y1:y2, x1:x2]
#     else:
#         y1, y2 = int(0.6 * h), int(0.8 * h)
#         x1, x2 = int(0.45 * w), int(0.8 * w)
#         offset_x, offset_y = x1, y1
#         roi = mask[y1:y2, x1:x2]
    

#     cv2.namedWindow("kernel observation", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("kernel observation", 800, 600)  # Set window size
#     cv2.imshow("kernel observation", roi)

#     # Find all contours in the mask
#     contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Prepare to save coordinates
#     coords = []

#     # Minimum area to filter out noise
#     if view != "Long View" and view != "Top View":
#         min_area = 500
#     else:
#         min_area = 2000
#     # max_area = 30000

#     #save corners
#     for contour in contours:
#         contour += [offset_x, offset_y]
#         hull = cv2.convexHull(contour)
#         area = cv2.contourArea(hull)

#         # Approximate contour to polygon to reduce points
#         if view == "Short View":
#             epsilon = 0.03 * cv2.arcLength(hull, True)  # adjust for precision
#         else:
#             epsilon = 0.01 * cv2.arcLength(hull, True)  # adjust for precision
#         approx = cv2.approxPolyDP(hull, epsilon, True)

#         if area > min_area:

#             corners = approx.reshape(-1, 2)  # shape (num_points, 2)

#             if len(corners) == 4:
#                 cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)  # Blue polygon outline
#                 for corner in corners:
#                     x, y = corner
#                     coords.append([x, y])

#     coords_one = sorted(coords, key=lambda x: x[0])[0:2]
#     coords_two = sorted(coords, key=lambda x: x[0])[2:]
#     coords_one = sorted(coords_one, key=lambda x: x[1])
#     coords_two = sorted(coords_two, key=lambda x: x[1])
#     coords = coords_one + coords_two
#     print(coords)

#     print(f"Detected {len(coords)} corners.")
#     if len(coords) < 4:
#         print("Error: Not enough corners detected. Please try again.")

#     # find remaining four points in the middle
#     if view == "Short View":
#         coords.append([(coords[2][0] + coords[3][0])/2, (coords[2][1] + coords[3][1])/2 - 25])
#         coords.append([(coords[2][0] + coords[3][0])/2, (coords[2][1] + coords[3][1])/2])
#         coords.append([(coords[0][0] + coords[1][0])/2, (coords[0][1] + coords[1][1])/2 - 25])
#         coords.append([(coords[0][0] + coords[1][0])/2, (coords[0][1] + coords[1][1])/2])
#     else:
#         coords.append([(coords[0][0] + coords[2][0])/2 - 10, (coords[0][1] + coords[2][1])/2])
#         coords.append([(coords[0][0] + coords[2][0])/2 + 10, (coords[0][1] + coords[2][1])/2])
#         coords.append([(coords[1][0] + coords[3][0])/2 - 10, (coords[1][1] + coords[3][1])/2])
#         coords.append([(coords[1][0] + coords[3][0])/2 + 10, (coords[1][1] + coords[3][1])/2])



#     #rearrange list
#     output = [[0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]
#     if view == "Short View":
#         output[0], output[1], output[2], output[3] = coords[0], coords[2], coords[4], coords[6]
#         output[4], output[5], output[6], output[7] = coords[7], coords[5], coords[3], coords[1]
#     else:
#         output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7] = coords[1], coords[6], coords[4], coords[0], coords[7], coords[3], coords[2], coords[5]

#     for out in output:
#         cv2.circle(frame, (int(out[0]), int(out[1])), 5, (0, 0, 255), -1)  # Red dots for corners

#     # # Save coordinates to a file
#     with open("selected_points.txt", "w") as f:
#         for x, y in output:
#             f.write(f"{x},{y}\n")

#     print(f"{len(output)} coordinates saved to file.")

#     # Show the result
#     cv2.namedWindow("Detected Yellow Rectangles", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Detected Yellow Rectangles", 800, 600)  # Set window size
#     cv2.imshow("Detected Yellow Rectangles", frame)

#     cv2.waitKey(0)  # Wait indefinitely until a key is pressed
#     cv2.destroyAllWindows()

#     return output

# if __name__ == "__main__":
#     cap = cv2.VideoCapture(r"C:\Users\Arnav\Downloads\JPJ_12_GS_long.vid01.MOV")
#     select_points(cap)