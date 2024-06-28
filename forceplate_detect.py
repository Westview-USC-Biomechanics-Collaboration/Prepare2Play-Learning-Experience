import PIL.Image
import cv2 as cv
import numpy
import numpy as np
from PIL import Image
from pixelmatch.contrib.PIL import pixelmatch

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
            boxCoords.append((col + xOffset, row + yOffset))
    return src


def CompareFrames(frame1: np.array, currentFrame: np.array, br: (), tl: ()):  #tl = tl_offset, br = br_corner

    croppedFrame1 = frame1[tl[1]: tl[1] + br[1], tl[0]:tl[0] + br[0]]
    croppedCurrentFrame = currentFrame[tl[1]: tl[1] + br[1], tl[0]:tl[0] + br[0]]

    f1 = PIL.Image.fromarray(croppedFrame1)
    c = PIL.Image.fromarray(croppedCurrentFrame)

    return pixelmatch(f1, c)


cam = cv.VideoCapture("data/5.5min_120Hz_SSRun_Fa19_OL_skele.mp4")

if not cam.isOpened():
    print("Unable to access the camera")
    cam.release()
    exit()


br_corner = (430, 25)
tl_offset = (768, 790)
frameCount = 0
while True:
    retrieved, frame = cam.read()
    frameCount += 1

    if frameCount == 1:
        initialFrame = frame
        cv.imwrite("forceplate_detection_img/5.5min_120Hz_SSRun_Fa19_OL_skele.png", frame)  #saves first frame

    if not retrieved:
        print("An error occurred, the stream has likely ended")
        break

    bounding = CreateBoundingBox(frame.copy(), br_corner, tl_offset)
    diff = CompareFrames(initialFrame, frame.copy(), br_corner, tl_offset)
    if diff > 1000:  # 100 is shadow, 1000 is foot
        print(frameCount)

    cv.imshow("Stream", bounding)

    if cv.waitKey(1) == ord("q"):  # gets the unicode value for q
        break

print(frameCount)
cam.release()
cv.destroyAllWindows()
