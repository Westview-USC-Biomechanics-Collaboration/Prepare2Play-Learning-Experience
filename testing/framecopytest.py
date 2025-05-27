import cv2

cam = cv2.VideoCapture(r"C:\Users\chase\Downloads\0519test.mp4")
while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break

    newFrame = frame.copy()  # Create a copy of the frame
    cv2.putText(newFrame, "This is a test", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Original Frame", frame)
    cv2.imshow("test", newFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
