import cv2 as cv
import numpy as np

# tl offset , br (from stream POV)
# br corner is assuming the tl is at (0,0) while the tl offset is where to put the (0,0) point
boxCoords = []


def CreateBoundingBox(src: np.array, br_corner: (), tl_offset: () = (0, 0)):
    global boxCoords
    xOffset = tl_offset[0]
    yOffset = tl_offset[1]
    for row in range(br_corner[1]):  # gets the y val of the br corner
        for col in range(br_corner[0]):  # gets the x val of the br corner
            src[row + yOffset][col + xOffset] = np.uint8([255, 0, 0])
            boxCoords.append((row + yOffset, col + xOffset))
    return src


def CompareFrames(frame1: np.array, currentFrame: np.array):
    for coord in boxCoords:
        x = coord[0]
        y = coord[1]

        if not (frame1[y][x] & currentFrame[y][x]).all():
            print("Hello")


cam = cv.VideoCapture("data/5.5min_120Hz_SSRun_Fa19_OL_skele.mp4")

if not cam.isOpened():
    print("Unable to access the camera")
    cam.release()
    exit()

frameCount = 0
while True:
    retrieved, frame = cam.read()
    frameCount += 1

    if frameCount == 1:
        cv.imwrite("forceplate_detection_img/5.5min_120Hz_SSRun_Fa19_OL_skele.png", frame)  #saves first frame

    if not retrieved:
        print("An error occurred, the stream has likely ended")
        break

    bounding = CreateBoundingBox(frame.copy(), (430, 25), (768, 790))
    CompareFrames(cv.imread("forceplate_detection_img/5.5min_120Hz_SSRun_Fa19_OL_skele.png"), frame.copy())
    cv.imshow("Stream", bounding)

    if cv.waitKey(1) == ord("q"):  # gets the unicode value for q
        break

print(frameCount)
cam.release()
cv.destroyAllWindows()
