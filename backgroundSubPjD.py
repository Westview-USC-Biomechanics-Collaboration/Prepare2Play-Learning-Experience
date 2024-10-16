import cv2
import numpy as np

# history = number of previous frames compared to to determine if a pixel is sufficiently "different"
# detectShadows = shadows in foreground
# varThreshhold = pixel variance limit to determine difference between foregound and background
# Low varThreshold: Small changes in pixel values are considered foreground. 
#                   This makes the algorithm more sensitive but might result in noise.
# High varThreshold: Larger deviations from the background are required for a pixel to be classified as foreground. 
#                    This reduces noise but may miss detecting smaller changes.

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history = 20, detectShadows = False, varThreshold = 500)

cap = cv2.VideoCapture('derenBasketballTest.mp4') 

lower_orange = np.array([10, 100, 100])
upper_orange = np.array([25, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (480, 270))

    fg_mask = bg_subtractor.apply(frame)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    orange_mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)

    combined_mask = cv2.bitwise_and(fg_mask, orange_mask)

    result_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)

    cv2.imshow('Original Frame', frame)
#    cv2.imshow('Foreground Mask (Background Subtraction)', fg_mask)
#    cv2.imshow('Orange Color Mask', orange_mask)
#    cv2.imshow('Combined Mask', combined_mask)
    cv2.imshow('Result Frame', result_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
