import PIL.Image
import cv2 as cv
import numpy as np
from pixelmatch.contrib.PIL import pixelmatch


class ForcePlateDetect:
    def __init__(self, videoPath):
        self.videoPath = videoPath
        self.cam = cv.VideoCapture(videoPath)
        self.firstFrame = self.getFirstFrame()
        self.currentFrame: np.ndarray = np.ndarray([])
        self.br_corner: () = None  # offsets are defined in pixel tuples
        self.tl_offset: () = None

    def getFirstFrame(self) -> np.ndarray:
        self.cam.set(cv.CAP_PROP_POS_FRAMES,
                     0)  # index starts at 0, https://stackoverflow.com/questions/11420748/setting-camera-parameters-in-opencv-python
        ret, frame = self.cam.read()
        if ret:
            return frame
        else:
            raise Exception("Unable to retrieve frame")

    # tl offset , br (from stream POV)
    # br corner is assuming the tl is at (0,0) while the tl offset is where to put the (0,0) point

    def CreateBoundingBox(self) -> np.array:  # Paints the box on the screen ; Helps to adjust the bounding area
        if (self.tl_offset is None) or (self.br_corner is None):
            raise Exception("tl_offset and br_corner are not defined")

        src = self.currentFrame.copy()

        xOffset = self.tl_offset[0]
        yOffset = self.tl_offset[1]
        for row in range(self.br_corner[1]):  # gets the y val of the br corner
            for col in range(self.br_corner[0]):  # gets the x val of the br corner
                src[row + yOffset][col + xOffset] = np.uint8([255, 0, 0])
        return src

    def CompareFrames(self) -> int:
        if (self.tl_offset is None) or (self.br_corner is None):
            raise Exception("tl_offset and br_corner are not defined")

        br = self.br_corner
        tl = self.tl_offset

        currentFrame = self.currentFrame.copy()

        croppedFrame1 = self.firstFrame[tl[1]: tl[1] + br[1], tl[0]:tl[0] + br[0]]
        croppedCurrentFrame = currentFrame[tl[1]: tl[1] + br[1], tl[0]:tl[0] + br[0]]

        f1 = PIL.Image.fromarray(croppedFrame1)
        c = PIL.Image.fromarray(croppedCurrentFrame)

        return pixelmatch(f1, c)

    def detect(self, tl_offset: (), br_corner: (), showView=False) -> tuple[int, float]:
        self.tl_offset = tl_offset
        self.br_corner = br_corner

        frameCount = 0
        print("Running the video...")
        while True:
            retrieved, frame = self.cam.read()
            self.currentFrame = frame
            frameCount += 1

            if not retrieved:
                print("An error occurred, the stream has likely ended")
                break

            diff = self.CompareFrames()

            if showView:
                bounding = self.CreateBoundingBox()
                cv.imshow("Stream", bounding)

            if diff > 1000:  # 100 is shadow, 1000 is foot
                break

            print(f"Checked Frame {frameCount}")

            if cv.waitKey(1) == ord("q"):  # gets the unicode value for q
                break

        fps = self.cam.get(cv.CAP_PROP_FPS)
        self.cam.release()
        cv.destroyAllWindows()
        return frameCount, fps
