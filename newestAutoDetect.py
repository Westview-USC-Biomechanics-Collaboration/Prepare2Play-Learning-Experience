import cv2
import numpy as np

def getYellowBallAndDrawOnFrame(video_path, output_txt="yellow_ball.txt"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    frame_count = 0
    display_frame = None

    while frame_count < 5:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count == 5:
            display_frame = frame.copy()
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Tuned HSV range for the yellow ball
            lower_yellow = np.array([18, 100, 150])
            upper_yellow = np.array([32, 255, 255])
            mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

            # Display mask at 75% scale
            cv2.imshow("Mask", cv2.resize(mask, None, fx=0.75, fy=0.75))
            cv2.waitKey(0)
            cv2.destroyWindow("Mask")

            # Apply morphological operations to reduce noise
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ball_center = None

            # Find the largest contour (assumed to be the ball)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 200:  # Ensure it's large enough to be the ball
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        ball_center = (cX, cY)

            # Draw on frame if ball is detected
            if ball_center and display_frame is not None:
                scale_percent = 50
                width = int(display_frame.shape[1] * scale_percent / 100)
                height = int(display_frame.shape[0] * scale_percent / 100)
                display_frame = cv2.resize(display_frame, (width, height))

                # Draw the ball center
                x_scaled = int(ball_center[0] * scale_percent / 100)
                y_scaled = int(ball_center[1] * scale_percent / 100)
                cv2.circle(display_frame, (x_scaled, y_scaled), 10, (0, 255, 0), -1)
                cv2.putText(display_frame, f"Ball: ({ball_center[0]}, {ball_center[1]})", (x_scaled + 10, y_scaled - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow("5th Frame with Yellow Ball", display_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # Save coordinates to text file
                with open(output_txt, "w") as f:
                    f.write(f"{ball_center[0]},{ball_center[1]}\n")

                return [ball_center]
            else:
                print("Warning: Yellow ball not detected.")
                return []

    cap.release()
    return []

# Example usage
if __name__ == "__main__":
    video_path = "testing\spk_JH_12_long_vid02 - Copy.MOV"
    try:
        ball_center = getYellowBallAndDrawOnFrame(video_path)
        print("Detected yellow ball center:", ball_center)
    except FileNotFoundError as e:
        print(e)