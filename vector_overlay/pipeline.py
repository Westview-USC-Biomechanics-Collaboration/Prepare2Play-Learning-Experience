import math

import cv2
import cv2 as cv
import pandas as pd
from test_corners import select_points
import numpy as np
import os

with open("example.in") as fin:
    data = fin.readline().split(" ")


def rect_to_trapezoid(x, y, rect_width, rect_height, trapezoid_coords):
    """
    Maps points from a rectangle to a trapezoid, simulating parallax distortion.

    Parameters:
    x, y: Coordinates of the point in the original rectangle (0 <= x <= rect_width, 0 <= y <= rect_height)
    rect_width, rect_height: Dimensions of the original rectangle
    trapezoid_coords: List of four (x, y) tuples representing the trapezoid corners in order:
                      [(top_left), (top_right), (bottom_right), (bottom_left)]

    Returns:
    new_x, new_y: Pixel coordinates of the mapped point in the trapezoid
    """
    # Ensure input coordinates are within the rectangle
    x = np.clip(x, 0, rect_width)
    y = np.clip(y, 0, rect_height)

    # Extract trapezoid coordinates
    (tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y) = trapezoid_coords

    # Calculate the left and right edge positions for the current y
    left_x = tl_x + (bl_x - tl_x) * (y / rect_height)
    right_x = tr_x + (br_x - tr_x) * (y / rect_height)

    # Calculate the width of the trapezoid at the current y
    trapezoid_width = right_x - left_x

    # Map x-coordinate
    new_x = left_x + (x / rect_width) * trapezoid_width

    # Calculate the top and bottom y positions of the trapezoid
    top_y = (tl_y + tr_y) / 2
    bottom_y = (bl_y + br_y) / 2

    # Map y-coordinate
    new_y = top_y + (y / rect_height) * (bottom_y - top_y)

    return (int(new_x), int(new_y))
class VectorOverlay:

    def __init__(self, fx1, fy1, fz1, ax1, ay1, fx2, fy2, fz2, ax2, ay2):
        self.fx1 = fx1
        self.fy1 = fy1
        self.fz1 = fz1
        self.px1 = ax1
        self.py1 = ay1
        self.fx2 = fx2
        self.fy2 = fy2
        self.fz2 = fz2
        self.px2 = ax2
        self.py2 = ay2

        self.frame_width = None
        self.frame_height = None
        self.fps = None
        self.frame_count = None

    def setFrameData(self):
        cap = cv.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        self.frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv.CAP_PROP_FPS))
        self.frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        print("####################################################")
        print(
            f"Frame width: {self.frame_width}, Frame height: {self.frame_height}, FPS: {self.fps}, Frame count: {self.frame_count}")
        cap.release()

    def normalizeForces(self):
        def normalize(input):
            output = math.log(abs(input+1))
            if input < 0:
                return -output
            else:
                return output

        self.fx1 = (normalize(self.fx1)*100/4) * self.frame_height
        self.fy1 = (normalize(self.fy1)*100/4) * self.frame_height
        self.fz1 = (normalize(self.fz1)*100/4) * self.frame_height
        self.ax1 = (normalize(self.ax1)*100/4) * self.frame_height
        self.ay1 = (normalize(self.ay1)*100/4) * self.frame_height
        self.fx2 = (normalize(self.fx2)*100/4) * self.frame_height
        self.fy2 = (normalize(self.fy2)*100/4) * self.frame_height
        self.fz2 = (normalize(self.fz2)*100/4) * self.frame_height
        self.ax2 = (normalize(self.ax2)*100/4) * self.frame_height
        self.ay2 = (normalize(self.ay2)*100/4) * self.frame_height

    def LongVectorOverlay(self, outputName):
        self.setFrameData(path=self.long_view_path)
        self.readData()
        self.normalizeForces(self.fy1, self.fy2, self.fz1, self.fz2)

        if self.frame_width is None or self.frame_height is None:
            print("Error: Frame data not set.")
            return

        if self.fz1 is None or self.fz2 is None:
            print("Error: Data not set.")
            return

        out = cv.VideoWriter(outputName, cv.VideoWriter_fourcc(*'mp4v'), self.fps,
                             (self.frame_width, self.frame_height))

        cap = cv.VideoCapture(self.long_view_path)
        frame_number = 0
        """
        plate 1                                 plate 2
        1   2   3   4       5   6   7   8   9   10  11  12  13      14  15
        Fx	Fy	Fz	|Ft|	Ax	Ay				Fx	Fy	Fz	|Ft|	Ax	Ay
        N	N	N	N	    m	m				N	N	N	 N	    m	m
        """
        self.check_corner(self.long_view_path, top=False)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # if this calls when the frame_number is equal to the total frame count then the stream has just ended
                print(f"Can't read frame at position {frame_number}")
                break

            fx1 = -self.fy1[int(frame_number)]
            fx2 = -self.fy2[int(frame_number)]
            fy1 = self.fz1[int(frame_number)]
            fy2 = self.fz2[int(frame_number)]
            py1 = self.px1[int(frame_number)]
            px1 = self.py1[int(frame_number)]
            py2 = self.px2[int(frame_number)]
            px2 = self.py2[int(frame_number)]
            # def drawArrows(self, frame, xf1, xf2, yf1, yf2, px1, px2, py1, py2):
            self.drawArrows(frame, fx1, fx2, fy1, fy2, px1, px2, py1, py2)
            cv2.imshow("window", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_number += 1
            out.write(frame)

        cap.release()
        out.release()
        print(f"Finished processing video; Total Frames: {frame_number}")

    def TopVectorOverlay(self, outputName):
        self.setFrameData(path=self.top_view_path)
        self.readData()
        self.normalizeForces(self.fy1, self.fy2, self.fx1, self.fx2)

        if self.frame_width is None or self.frame_height is None:
            print("Error: Frame data not set.")
            return

        if self.fz1 is None or self.fz2 is None:
            print("Error: Data not set.")
            return

        out = cv.VideoWriter(outputName, cv.VideoWriter_fourcc(*'mp4v'), self.fps,
                             (self.frame_width, self.frame_height))

        cap = cv.VideoCapture(self.top_view_path)
        frame_number = 0
        self.check_corner(self.top_view_path, top=True)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # if this calls when the frame_number is equal to the total frame count then the stream has just ended
                print(f"Can't read frame at position {frame_number}")
                break

            fx1 = -self.fy1[int(frame_number)]
            fx2 = -self.fy2[int(frame_number)]
            fy1 = -self.fx1[int(frame_number)]
            fy2 = -self.fx2[int(frame_number)]
            py1 = self.px1[int(frame_number)]
            px1 = self.py1[int(frame_number)]
            py2 = self.px2[int(frame_number)]
            px2 = self.py2[int(frame_number)]
            # print(f"x:{py1}, y:{py2}\n")

            self.drawArrows(frame, fx1, fx2, fy1, fy2, px1, px2, py1, py2)
            cv2.imshow("window", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_number += 1
            out.write(frame)

        cap.release()
        out.release()
        print(f"Finished processing video; Total Frames: {frame_number}")
    def ShortVectorOverlay(self, outputName):
        self.setFrameData(path=self.short_view_path)
        self.readData()
        self.normalizeForces([0], self.fx2, [0], self.fz2)

        if self.frame_width is None or self.frame_height is None:
            print("Error: Frame data not set.")
            return

        if self.fz1 is None or self.fz2 is None:
            print("Error: Data not set.")
            return

        out = cv.VideoWriter(outputName, cv.VideoWriter_fourcc(*'mp4v'), self.fps,
                             (self.frame_width, self.frame_height))

        cap = cv.VideoCapture(self.short_view_path)
        frame_number = 0
        self.check_corner(self.short_view_path, top=False)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # if this calls when the frame_number is equal to the total frame count then the stream has just ended
                print(f"Can't read frame at position {frame_number}")
                break
            # This only shows the force on force plate 2, you can adjust this part so that it shows the force on force plate 1
            fx1 = 0
            fx2 = self.fx2[int(frame_number)]
            fy1 = 0
            fy2 = self.fz2[int(frame_number)]
            py1 = self.px1[int(frame_number)]
            px1 = 1 - self.py1[int(frame_number)]
            py2 = self.py2[int(frame_number)]
            px2 = 1 - self.px2[int(frame_number)]
            self.drawArrows(frame, fx1, fx2, fy1, fy2, px1, px2, py1, py2)
            cv2.imshow("window", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_number += 1
            out.write(frame)

        cap.release()
        out.release()
        print(f"Finished processing video; Total Frames: {frame_number}")

    def drawArrows(self, frame, xf1, xf2, yf1, yf2, px1, px2, py1, py2):

        start_point_1 = rect_to_trapezoid(px1, py1, 1, 1,
                                          [self.corners[0], self.corners[1], self.corners[2], self.corners[3]])
        start_point_2 = rect_to_trapezoid(px2, py2, 1, 1,
                                          [self.corners[4], self.corners[5], self.corners[6], self.corners[7]])
        # print(f"Startpoint1: {start_point_1}, Startpoint2:{start_point_2}")

        end_point_1 = (int(start_point_1[0] + xf1), int(start_point_1[1] - yf1))
        end_point_2 = (int(start_point_2[0] + xf2), int(start_point_2[1] - yf2))
        # print(start_point_1)
        # print(end_point_1)
        #
        # print(start_point_2)
        # print(end_point_2)
        cv.arrowedLine(frame, start_point_1, end_point_1, (0, 255, 0), 2)

        cv.arrowedLine(frame, start_point_2, end_point_2, (255, 0, 0), 2)

        # Draw red dots for centers
        # cv.circle(frame, self.corners[0], 5, (0, 0, 255), -1)  # Red dot at start_point_1
        # cv.circle(frame, self.corners[1], 5, (0, 0, 255), -1)  # Red dot at end_point_1
        # cv.circle(frame, self.corners[2], 5, (0, 0, 255), -1)  # Red dot at start_point_2
        # cv.circle(frame, self.corners[3], 5, (0, 0, 255), -1)  # Red dot at end_point_2
        # cv.circle(frame, self.corners[4], 5, (0, 0, 255), -1)  # Red dot at start_point_1
        # cv.circle(frame, self.corners[5], 5, (0, 0, 255), -1)  # Red dot at end_point_1
        # cv.circle(frame, self.corners[6], 5, (0, 0, 255), -1)  # Red dot at start_point_2
        # cv.circle(frame, self.corners[7], 5, (0, 0, 255), -1)  # Red dot at end_point_2