import cv2 as cv
import numpy as np


def BGR2HSV(color: []):
    col = np.uint8([[color]])  # pixel format
    return cv.cvtColor(col, cv.COLOR_BGR2HSV)


# tl offset , br (from stream POV)
# br corner is assuming the tl is at (0,0) while the tl offset is where to put the (0,0) point
def CreateBoundingBox(src: np.array, br_corner: (), tl_offset: () = (0, 0)):
    xOffset = tl_offset[0]
    yOffset = tl_offset[1]
    for row in range(br_corner[1]):  # gets the y val of the br corner
        for col in range(br_corner[0]):  #gets the x val of the br corner
            src[row + yOffset][col + xOffset] = np.uint8([0, 255, 0])
            # if src[row][col] != np.uint8([0, 255, 0]):
            #     print("hello word")

    return src


color = BGR2HSV([255, 127, 80])
print(color)

lower_bound = np.array([20, 110, 110])
upper_bound = np.array([40, 255, 255])

cam = cv.VideoCapture(0)

if not cam.isOpened():
    print("Unable to access the camera")
    cam.release()
    exit()

while True:
    retrieved, frame = cam.read()
    if not retrieved:
        print("An error occurred, the stream has likely ended")
        break

    # frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # mask = cv.inRange(frame_HSV, lower_bound, upper_bound) # b&w img
    # coloredMask = cv.bitwise_and(frame, frame, mask=mask)
    #
    # ret, thresh = cv.threshold(mask, 200, 255, cv.THRESH_BINARY) #https://stackoverflow.com/questions/44378099/opencv-draw-contours-of-objects-in-the-binary-image
    # contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # cv.drawContours(frame, contours, -1, (0, 255, 0), 3)
    bounding = CreateBoundingBox(frame, (100, 200), (200, 100))
    cv.imshow("Stream", bounding)

    if cv.waitKey(1) == ord("q"):  # gets the unicode value for q
        break

cam.release()
cv.destroyAllWindows()
