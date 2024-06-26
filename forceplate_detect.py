import cv2 as cv
import numpy as np


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


cam = cv.VideoCapture("data/5.5min_120Hz_SSRun_Fa19_OL_skele.mp4")

if not cam.isOpened():
    print("Unable to access the camera")
    cam.release()
    exit()

while True:
    retrieved, frame = cam.read()
    if not retrieved:
        print("An error occurred, the stream has likely ended")
        break

    bounding = CreateBoundingBox(frame, (100, 200), (200, 100))
    cv.imshow("Stream", bounding)

    if cv.waitKey(1) == ord("q"):  # gets the unicode value for q
        break

cam.release()
cv.destroyAllWindows()
