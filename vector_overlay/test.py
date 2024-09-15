import cv2


def main():
    # Open the webcam (index 0 is usually the default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    current_frame = 0
    print("Press the arrow keys to move frames, 'Esc' to exit.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the frame in a window
        cv2.imshow("Webcam", frame)

        # Wait for a key press
        key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press

        if key == 27:  # Press 'Esc' to exit
            print("Esc pressed, exiting...")
            break
        elif key == ord('d'):  # Right arrow (next frame)
            current_frame += 1
            print(f"Right arrow pressed, next frame: {current_frame}")
        elif key == ord('a'):  # Left arrow (previous frame)
            current_frame -= 1
            print(f"Left arrow pressed, previous frame: {current_frame}")
        else:
            # Print unhandled key press
            print(f"Unhandled key pressed: {key}")

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
