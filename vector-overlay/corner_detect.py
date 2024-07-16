import cv2 as cv
from enum import Enum
import numpy as np


class Views(Enum):
    Side = 0
    Top = 1


class FindCorners:
    def __init__(self, videoPath):
        self.videoPath = videoPath

    def find(self, view: Views):
        if view.value == 0:
            cap = cv.VideoCapture(self.videoPath)
            print(cv.cvtColor(np.uint8([[[0, 58, 173]]]), cv.COLOR_BGR2HSV))
            lower_color = np.array([100, 230, 20])
            upper_color = np.array([200, 275, 200])

            if not cap.isOpened():
                print("Unable to open camera")
                cap.release()
                exit()

            while True:
                ret, frame = cap.read()

                # frame = frame[800:, 400:]
                # frame = frame[:, :1000]

                if not ret:
                    print("Can't find frame")
                    break
                hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

                colorMask = cv.inRange(hsvFrame, lower_color, upper_color)
                # result = cv.bitwise_and(frame, frame, mask=colorMask)
                cleanedUpImg = cv.erode(colorMask, np.ones((5, 5), np.uint8), iterations=1)

                ret, thresh = cv.threshold(cleanedUpImg, 0, 255, 0)

                contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                cv.drawContours(cleanedUpImg, contours, -1, (0, 255, 0), 3)

                cv.imshow("frame", cleanedUpImg)

                if cv.waitKey(1) == ord("q"):  # gets the unicode value for q
                    cap.release()
                    exit()
