import cv2 as cv
from enum import Enum
import numpy as np


class Views(Enum):
    Side = 0
    Top = 1


class FindCorners:
    def __init__(self, videoPath):
        self.videoPath = videoPath
        self.videoFPS = cv.VideoCapture(videoPath).get(cv.CAP_PROP_FRAME_COUNT)

    def find(self, view: Views):
        if view.value == 0:
            cap = cv.VideoCapture(self.videoPath)
            print(cv.cvtColor(np.uint8([[[0, 58, 173]]]), cv.COLOR_BGR2HSV))
            lower_color = np.array([95, 200, 10])
            upper_color = np.array([210, 255, 220])

            if not cap.isOpened():
                print("Unable to open camera")
                cap.release()
                exit()
            frameNum = 0
            while True:
                frameNum += 1
                ret, frame = cap.read()

                if not ret:
                    print("Can't find frame")
                    break

                topOffset = 900
                leftOffset = 400
                frame = frame[topOffset:, leftOffset:]
                frame = frame[:, :1000]

                hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

                colorMask = cv.inRange(hsvFrame, lower_color, upper_color)
                # result = cv.bitwise_and(frame, frame, mask=colorMask)
                cleanedUpImg = cv.erode(colorMask, np.ones((5, 5), np.uint8), iterations=1)

                ret, thresh = cv.threshold(cleanedUpImg, 127, 255, 0)

                contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                # cv.drawContours(frame, contours, -1, (0, 255, 0), 3)
                # cv.imshow("frame", frame)

                if frameNum == self.videoFPS:
                    markerList = []
                    for contour in contours:
                        M = cv.moments(contour)
                        cx = int(M['m10'] / M['m00']) + leftOffset
                        cy = int(M['m01'] / M['m00']) + topOffset
                        markerList.append((cx, cy))
                    # once we have the list, we can trim off the bottom points as they aren't needed
                    relevantPoints = sorted(markerList, key = lambda x: x[1])[0:4] # all the top points
                    cornerList = sorted(relevantPoints, key = lambda x: x[0])
                    # cornerList is 4 points [tl_1, tr_1, tl_2, tr_2]
                    return cornerList

                # if cv.waitKey(1) == ord("q"):  # gets the unicode value for q
                #     cap.release()
                #     exit()
