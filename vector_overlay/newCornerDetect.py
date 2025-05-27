import cv2
import numpy as np
import os

def detect_blue_corners(video_path, output_txt="blue_corners.txt", plate_length_inches=24,
                        custom_x5=50, custom_y5=50,
                        custom_x8=50, custom_y8=50):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    blue_corners = []
    frame_with_corners = None

    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    while len(blue_corners) < 4:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_frame_corners = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    current_frame_corners.append((cX, cY))

        for pt in current_frame_corners:
            if all(np.linalg.norm(np.array(pt) - np.array(existing)) > 20 for existing in blue_corners):
                blue_corners.append(pt)
                if len(blue_corners) == 4:
                    frame_with_corners = frame.copy()
                    break

    cap.release()

    blue_corners = sorted(blue_corners, key=lambda pt: (pt[0], pt[1]))

    exaggeration_factor = 15.0

    if len(blue_corners) >= 2:
        y_dist = abs(blue_corners[3][1] - blue_corners[0][1])
        pixel_per_inch = (y_dist / plate_length_inches) * exaggeration_factor
    else:
        pixel_per_inch = 10 * exaggeration_factor

    y_offset = int(plate_length_inches * pixel_per_inch)

    # Infer points 6 and 7 from original 2 and 3
    inferred_6 = (blue_corners[1][0], blue_corners[1][1] - y_offset)
    inferred_7 = (blue_corners[2][0], blue_corners[2][1] - y_offset)

    # Use custom inputs for points 5 and 8
    inferred_5 = (custom_x5, custom_y5)
    inferred_8 = (custom_x8, custom_y8)

    # Original order: [1, 2, 3, 4, 5, 6, 7, 8]
    all_corners = blue_corners + [inferred_5, inferred_6, inferred_7, inferred_8]

    # New order: 5, 6, 2, 1, 7, 8, 4, 3 (index-wise: 4, 5, 1, 0, 6, 7, 3, 2)
    display_order = [4, 5, 1, 0, 6, 7, 3, 2]
    ordered_corners = [all_corners[i] for i in display_order]

    # Save in new order
    with open(output_txt, "w") as f:
        for (x, y) in ordered_corners:
            f.write(f"{x}, {y}\n")
    print(f"Saved reordered corners to {output_txt}")

    # Draw numbers in new order (from 1 to 8)
    if frame_with_corners is not None:
        for idx, (x, y) in enumerate(ordered_corners):
            cv2.circle(frame_with_corners, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(frame_with_corners, str(idx + 1), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("Detected + Reordered Blue Corners", frame_with_corners)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
detect_blue_corners("example.mov", plate_length_inches=24, custom_x5=515, custom_y5=900, custom_x8=1340, custom_y8=912)
