# Import libraries
import cv2
import cv2 as cv
import pandas as pd
from vector_overlay.select_corners import select_points
import numpy as np
import os

"""
This is the vector overlay code for GUI
The difference is the input.

plate 1                                 plate 2
1   2   3   4       5   6   7   8   9   10  11  12  13      14  15
Fx	Fy	Fz	|Ft|	Ax	Ay				Fx	Fy	Fz	|Ft|	Ax	Ay
N	N	N	N	    m	m				N	N	N	 N	    m	m

force plate directions(at top view)
(0.0)_______________________________________
    |                                      |
    |                 Ax                   |
    |                  |                   |
    |                  |                   |
    |          Fy---------------Ay         |
    |                  |                   |
    |                  |                   |
    |                  Fx                  |
    |______________________________________|

Select corner sequence:
    long view/ side view:
       1              2 5          6
      /---------------||----------\\
     /               | |            \\
   4/---------------|3 8|-------------\\7
    |---------------|   |--------------|

    top view:
    1               2  5                6
     _______________    ________________
    |               |  |                |
    |               |  |                |
    |               |  |                |
    ----------------   ----------------- 
    4               3  8                7

    shortview/ front view:
                1        2
                __________
               /          \\
              /            \\
          4  |--------------| 3
           5__________________  6
           /                  \\
          /                    \\
       8/______________________\\ 7     
         |_______________________|

"""
def rect_to_trapezoid(x, y, rect_width, rect_height, trapezoid_coords,short = False):
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
    left_x = bl_x + (tl_x - bl_x) * y
    right_x = br_x + (tr_x - br_x) * y

    # Calculate the width of the trapezoid at the current y
    trapezoid_width = right_x - left_x

    # Map x-coordinate
    new_x = left_x + x * trapezoid_width

    # Calculate the top and bottom y positions of the trapezoid
    top_y = (tl_y + tr_y) / 2
    bottom_y = (bl_y + br_y) / 2

    # Map y-coordinate
    if short:
        new_y = bottom_y + y * (top_y - bottom_y)
    else:
        new_y = top_y + y * (bottom_y - top_y)

    return (int(new_x), int(new_y))

class VectorOverlay:

    def __init__(self, data, video):
        # data is a pandas dataframe, orientation is either "top" "long" "short"
        self.data = data
        self.video = video



        self.frame_width = None
        self.frame_height = None
        self.fps = None
        self.frame_count = None

        self.fx1 = ()
        self.fy1 = ()
        self.fz1 = ()
        self.px1 = ()
        self.py1 = ()

        self.fx2 = ()
        self.fy2 = ()
        self.fz2 = ()
        self.px2 = ()
        self.py2 = ()

        self.corners = []

        # initializing
        self.setFrameData()
        self.check_corner(cap=self.video)
        self.readData()

    def check_corner(self, cap):
        self.corners = select_points(cap=cap)

    def setFrameData(self):
        cap = self.video

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        self.frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv.CAP_PROP_FPS))
        self.frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        print(
            f"Frame width: {self.frame_width}, Frame height: {self.frame_height}, FPS: {self.fps}, Frame count: {self.frame_count}")

    def normalizeForces(self, x1, x2, y1, y2):
        max_force = max(
            max(abs(value) for value in x1),
            max(abs(value) for value in x2),
            max(abs(value) for value in y1),
            max(abs(value) for value in y2)
        )
        scale_factor = min(self.frame_height, self.frame_width) * 0.8 / max_force

        self.fx1 = tuple(f * scale_factor for f in self.fx1)
        self.fy1 = tuple(f * scale_factor for f in self.fy1)
        self.fz1 = tuple(f * scale_factor for f in self.fz1)
        self.fx2 = tuple(f * scale_factor for f in self.fx2)
        self.fy2 = tuple(f * scale_factor for f in self.fy2)
        self.fz2 = tuple(f * scale_factor for f in self.fz2)

    def readData(self):
        frame_count = self.frame_count
        step_size = 10 # should be 10
        current_row = 0
        fx1 = []
        fy1 = []
        fz1 = []
        px1 = []
        py1 = []

        fx2 = []
        fy2 = []
        fz2 = []
        px2 = []
        py2 = []
        for i in range(frame_count):
            start_row = int(round(current_row))
            end_row = int(round(current_row + step_size))
            current_row += step_size

            # Check if start_row exceeds the DataFrame length
            if start_row >= len(self.data):
                # If start_row is out of bounds, append default values (0.0)
                data_x1 = 0.0
                data_y1 = 0.0
                data_z1 = 0.0
                pressure_x1 = 0.0
                pressure_y1 = 0.0

                data_x2 = 0.0
                data_y2 = 0.0
                data_z2 = 0.0
                pressure_x2 = 0.0
                pressure_y2 = 0.0
            else:
                # Normal data extraction if start_row is within bounds
                data_x1 = self.data.loc[start_row, "Fx1"] if not pd.isna(self.data.loc[start_row, "Fx1"]) else 0.0
                data_y1 = self.data.loc[start_row, "Fy1"] if not pd.isna(self.data.loc[start_row, "Fy1"]) else 0.0
                data_z1 = self.data.loc[start_row, "Fz1"] if not pd.isna(self.data.loc[start_row, "Fz1"]) else 0.0
                pressure_x1 = (self.data.loc[start_row, "Ax1"] + 0.3) / 0.6 if not pd.isna(
                    self.data.loc[start_row, "Ax1"]) else 0.0
                pressure_y1 = (self.data.loc[start_row, "Ay1"] + 0.45) / 0.9 if not pd.isna(
                    self.data.loc[start_row, "Ay1"]) else 0.0

                data_x2 = self.data.loc[start_row, "Fx2"] if not pd.isna(self.data.loc[start_row, "Fx2"]) else 0.0
                data_y2 = self.data.loc[start_row, "Fy2"] if not pd.isna(self.data.loc[start_row, "Fy2"]) else 0.0
                data_z2 = self.data.loc[start_row, "Fz2"] if not pd.isna(self.data.loc[start_row, "Fz2"]) else 0.0
                pressure_x2 = (self.data.loc[start_row, "Ax2"] + 0.3) / 0.6 if not pd.isna(
                    self.data.loc[start_row, "Ax2"]) else 0.0
                pressure_y2 = (self.data.loc[start_row, "Ay2"] + 0.45) / 0.9 if not pd.isna(
                    self.data.loc[start_row, "Ay2"]) else 0.0

            # Append the data to the lists
            fx1.append(data_x1)
            fy1.append(data_y1)
            fz1.append(data_z1)
            px1.append(pressure_x1)
            py1.append(pressure_y1)

            fx2.append(data_x2)
            fy2.append(data_y2)
            fz2.append(data_z2)
            px2.append(pressure_x2)
            py2.append(pressure_y2)

        self.fx1 = tuple(fx1)
        self.fy1 = tuple(fy1)
        self.fz1 = tuple(fz1)
        self.px1 = tuple(px1)
        self.py1 = tuple(py1)

        self.fx2 = tuple(fx2)
        self.fy2 = tuple(fy2)
        self.fz2 = tuple(fz2)
        self.px2 = tuple(px2)
        self.py2 = tuple(py2)

    def drawArrows(self, frame, xf1, xf2, yf1, yf2, px1, px2, py1, py2,short=False):
        if short:
            point_pair1 = rect_to_trapezoid(px1, py1, 1, 1,
                                            [self.corners[0], self.corners[1], self.corners[2], self.corners[3]],short=True)

            point_pair2 = rect_to_trapezoid(px2, py2, 1, 1,
                                            [self.corners[4], self.corners[5], self.corners[6], self.corners[7]],short=True)

        else:
            point_pair1 = rect_to_trapezoid(px1, py1, 1, 1,
                                              [self.corners[0], self.corners[1], self.corners[2], self.corners[3]])

            point_pair2 = rect_to_trapezoid(px2, py2, 1, 1,
                                              [self.corners[4], self.corners[5], self.corners[6], self.corners[7]])


        end_point_1 = (int(point_pair1[0] + xf1), int(point_pair1[1] - yf1))
        end_point_2 = (int(point_pair2[0] + xf2), int(point_pair2[1] - yf2))

        cv.arrowedLine(frame, point_pair1, end_point_1, (0, 255,0), 4)

        cv.arrowedLine(frame, point_pair2, end_point_2, (255, 0, 0), 4)

    def LongVectorOverlay(self, outputName):
        self.normalizeForces(self.fy1, self.fy2, self.fz1, self.fz2)

        if self.frame_width is None or self.frame_height is None:
            print("Error: Frame data not set.")
            return

        if self.fz1 is None or self.fz2 is None:
            print("Error: Data not set.")
            return

        out = cv.VideoWriter(outputName, cv.VideoWriter_fourcc(*'mp4v'), self.fps,
                             (self.frame_width, self.frame_height))


        frame_number = 0
        """
        plate 1                                 plate 2
        1   2   3   4       5   6   7   8   9   10  11  12  13      14  15
        Fx	Fy	Fz	|Ft|	Ax	Ay				Fx	Fy	Fz	|Ft|	Ax	Ay
        N	N	N	N	    m	m				N	N	N	 N	    m	m
        """
        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                # if this calls when the frame_number is equal to the total frame count then the stream has just ended
                print(f"Can't read frame at position {frame_number}")
                break

            # in all methods, we have fx and fy, that doesn't mean the actual fx and fy in force data
            # Instead, that's the fx and fy in our video view. x and y direction

            fx1 = -self.fy1[int(frame_number)]
            fx2 = -self.fy2[int(frame_number)]
            fy1 = self.fz1[int(frame_number)]
            fy2 = self.fz2[int(frame_number)]

            px1 = self.py1[int(frame_number)]
            py1 = self.px1[int(frame_number)]
            px2 = self.py2[int(frame_number)]
            py2 = self.px2[int(frame_number)]

            self.drawArrows(frame, fx1, fx2, fy1, fy2, px1, px2, py1, py2)
            cv2.imshow("window", cv2.resize(frame, (int(self.frame_width * 0.5), int(self.frame_height * 0.5)))
)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_number += 1
            out.write(frame)


        out.release()
        cv2.destroyAllWindows()
        print(f"Finished processing video; Total Frames: {frame_number}")

    def TopVectorOverlay(self, outputName):
        self.normalizeForces(self.fy1, self.fy2, self.fx1, self.fx2)

        if self.frame_width is None or self.frame_height is None:
            print("Error: Frame data not set.")
            return

        if self.fz1 is None or self.fz2 is None:
            print("Error: Data not set.")
            return

        out = cv.VideoWriter(outputName, cv.VideoWriter_fourcc(*'mp4v'), self.fps,
                             (self.frame_width, self.frame_height))

        cap = self.video
        frame_number = 0

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

            px1 = self.py1[int(frame_number)]
            py1 = 1-self.px1[int(frame_number)]  # I don't know why, but the real px is not the same as I assumed
            px2 = self.py2[int(frame_number)]
            py2 = 1-self.px2[int(frame_number)]

            self.drawArrows(frame, fx1, fx2, fy1, fy2, px1, px2, py1, py2)
            cv2.imshow("window", cv2.resize(frame, (int(self.frame_width * 0.5), int(self.frame_height * 0.5))))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_number += 1
            out.write(frame)

        out.release()
        cv2.destroyAllWindows()
        print(f"Finished processing video; Total Frames: {frame_number}")

    # short view need more test
    def ShortVectorOverlay(self, outputName):
        self.normalizeForces([0], self.fx2, [0], self.fz2)

        if self.frame_width is None or self.frame_height is None:
            print("Error: Frame data not set.")
            return

        if self.fz1 is None or self.fz2 is None:
            print("Error: Data not set.")
            return

        out = cv.VideoWriter(outputName, cv.VideoWriter_fourcc(*'mp4v'), self.fps,
                             (self.frame_width, self.frame_height))

        cap = self.video
        frame_number = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # if this calls when the frame_number is equal to the total frame count then the stream has just ended
                print(f"Can't read frame at position {frame_number}")
                break
            if(self.corners[0][1]<self.corners[4][1]):
                print("force plate 2 in front")
                # This only shows the force on force plate 2, you can adjust this part so that it shows the force on force plate 1
                fx1 = -self.fx1[int(frame_number)]
                fx2 = -self.fx2[int(frame_number)]
                fy1 = self.fz1[int(frame_number)]
                fy2 = self.fz2[int(frame_number)]
                px1 = self.px1[int(frame_number)]
                px2 = self.px2[int(frame_number)]
                py1 = 1 - self.py1[int(frame_number)]
                py2 = 1 - self.py2[int(frame_number)]
            else:
                print("force plate 1 in front")
                fx1 = self.fx1[int(frame_number)]
                fx2 = self.fx2[int(frame_number)]
                fy1 = self.fz1[int(frame_number)]
                fy2 = self.fz2[int(frame_number)]
                px1 = 1 - self.px1[int(frame_number)]
                px2 = 1 - self.px2[int(frame_number)]
                py1 = 1 - self.py1[int(frame_number)]
                py2 = 1 - self.py2[int(frame_number)]

            self.drawArrows(frame, fx1, fx2, fy1, fy2, px1, px2, py1, py2,short=True)
           # Resize the frame for display
            resized_frame = cv2.resize(frame, (int(self.frame_width * 0.5), int(self.frame_height * 0.5)))

            # Show the resized frame
            cv2.imshow("window", resized_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_number += 1
            out.write(frame)

        out.release()
        cv2.destroyAllWindows()
        print(f"Finished processing video; Total Frames: {frame_number}")

if __name__ == "__main__":
    df = pd.read_excel("C:\\Users\\16199\Desktop\data\Chase\\bcp_lr_CC_for02_Raw_Data.xlsx",skiprows=19)
    cap = cv2.VideoCapture("C:\\Users\\16199\Desktop\data\Chase\\bcp_lr_CC_vid02.mp4")
    v = VectorOverlay(df,cap)
    v.LongVectorOverlay(outputName="C:\\Users\\16199\Desktop\data\Chase\\testoutput.mp4")
