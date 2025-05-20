import cv2
import numpy as np

def getBlueCentersAndDrawOnFrame(video_path, distance_threshold=50):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    frame_count = 0
    raw_points = []
    display_frame = None

    while frame_count < 5:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count == 5:
            display_frame = frame.copy()  # Save frame for display
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_blue = np.array([100, 150, 50])
            upper_blue = np.array([140, 255, 255])
            mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    raw_points.append((cX, cY))

    cap.release()

    # Group and average points
    clustered_centers = []
    used = [False] * len(raw_points)

    for i, (x1, y1) in enumerate(raw_points):
        if used[i]:
            continue
        group = [(x1, y1)]
        used[i] = True

        for j in range(i + 1, len(raw_points)):
            if used[j]:
                continue
            x2, y2 = raw_points[j]
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist <= distance_threshold:
                group.append((x2, y2))
                used[j] = True

        if group:
            avg_x = int(np.mean([pt[0] for pt in group]))
            avg_y = int(np.mean([pt[1] for pt in group]))
            clustered_centers.append((avg_x, avg_y))

            # Draw the center on the frame
            cv2.circle(display_frame, (avg_x, avg_y), 10, (0, 255, 0), -1)
            cv2.putText(display_frame, f"({avg_x}, {avg_y})", (avg_x + 10, avg_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show the frame with markers
    if display_frame is not None:
        scale_percent = 50  # Resize to 50%
        width = int(display_frame.shape[1] * scale_percent / 100)
        height = int(display_frame.shape[0] * scale_percent / 100)
        resized_frame = cv2.resize(display_frame, (width, height))

        cv2.imshow("5th Frame with Blue Centers", resized_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return clustered_centers

# Example usage
if __name__ == "__main__":
    video_path = "testing/ajp_lr_JN_long_vid.05.mov"
    try:
        blue_centers = getBlueCentersAndDrawOnFrame(video_path)
        print("Detected blue centers:", blue_centers)
    except FileNotFoundError as e:
        print(e)
