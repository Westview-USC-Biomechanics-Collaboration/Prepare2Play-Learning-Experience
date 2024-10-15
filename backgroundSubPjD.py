import cv2

# history = number of previous frames compared to to determine if a pixel is sufficiently "different"
# detectShadows = shadows in foreground
# varThreshhold = pixel variance limit to determine difference between foregound and background
# Low varThreshold: Small changes in pixel values are considered foreground. 
#                   This makes the algorithm more sensitive but might result in noise.
# High varThreshold: Larger deviations from the background are required for a pixel to be classified as foreground. 
#                    This reduces noise but may miss detecting smaller changes.

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history = 20, detectShadows = False, varThreshold = 500)

cap = cv2.VideoCapture("derenBasketballTest.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    resized_frame = cv2.resize(frame, (480, 270))

    fg_mask = bg_subtractor.apply(resized_frame)

    cv2.imshow('Original Frame', resized_frame)
    cv2.imshow('Foreground Mask', fg_mask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()