import cv2

# Open the default camera (webcam)
camera = cv2.VideoCapture(1)

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop to continuously capture frames
while True:
    # Capture a frame
    ret, frame = camera.read()

    # Check if frame is captured successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
