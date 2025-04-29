import cv2

cap = cv2.VideoCapture(0)  # or the correct device index for your capture card
cap.set(cv2.CAP_PROP_FPS, 240)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Process the frame here
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
