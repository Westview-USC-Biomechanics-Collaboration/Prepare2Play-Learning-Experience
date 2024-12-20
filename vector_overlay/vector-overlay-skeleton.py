# Import libraries
import cv2
import cv2 as cv
import pandas as pd
from test_corners import select_points
import numpy as np
import os
"""
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


# Initialization
"""
set your output path here
"""
def outputname(path):
    """
    use "\\" if you are in windows
    use "/" if you are in ios
    """
    filename = path.split("\\")[-1][:-4]
    output_name = "outputs\\" + filename + "_vector_overlay.mp4"
    return output_name

def find_files(directory):
    long_file = None
    short_file = None
    top_file = None
    data = None

    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            if "long" in filename:
                long_file = os.path.join(directory, filename)
            elif "short" in filename:
                short_file = os.path.join(directory, filename)
            elif "top" in filename:
                top_file = os.path.join(directory, filename)
            else:
                long_file = os.path.join(directory, filename)
        elif filename.endswith(".xlsx"):
            data = os.path.join(directory, filename)

    return long_file, short_file, top_file, data

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
    left_x = bl_x + (tl_x - bl_x) * (y / rect_height)
    right_x = br_x + (tr_x - br_x) * (y / rect_height)

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

    def __init__(self, top_view_path, long_view_path, short_view_path, data_path):
        self.top_view_path = top_view_path
        self.long_view_path = long_view_path
        self.short_view_path = short_view_path
        self.data_path = data_path
        names = ["abs time (s)", "Fx1", "Fy1", "Fz1", "|Ft1|", "Ax1", "Ay1", "COM px1", "COM py1", "COM pz1",
                 "Fx2", "Fy2", "Fz2", "|Ft2|", "Ax2", "Ay2", "COM px2", "COM py2", "COM pz2"]
        df = pd.read_excel(
            self.data_path
        )
        data = df.iloc[18:, 0:len(names)].reset_index(drop=True)
        data.columns = names
        self.data = data

        self.frame_width, self.frame_height, self.fps, self.frame_count = None, None, None, None

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

    def check_corner(self, path, top=False):
        self.corners = select_points(video_path=path, top=top)

    def check_direction(self, points):
        # Assuming points is a list of tuples [(x1, y1), (x2, y2)]
        if points[0][0] > points[1][0]:  # Compare the x-coordinates
            return True
        return False

    def setFrameData(self, path):
        print(f"Opening video: {path}")
        cap = cv.VideoCapture(path)

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        self.frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv.CAP_PROP_FPS))
        self.frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        print(
            f"Frame width: {self.frame_width}, Frame height: {self.frame_height}, FPS: {self.fps}, Frame count: {self.frame_count}")
        cap.release()

    def normalizeForces(self, x1, x2, y1, y2):
        max_force = max(
            max(abs(value) for value in x1),
            max(abs(value) for value in x2),
            max(abs(value) for value in y1),
            max(abs(value) for value in y2)
        )
        scale_factor = min(self.frame_height,self.frame_width)*0.8 / max_force
        
        self.fx1 = tuple(f * scale_factor for f in self.fx1)
        self.fy1 = tuple(f * scale_factor for f in self.fy1)
        self.fz1 = tuple(f * scale_factor for f in self.fz1)
        self.fx2 = tuple(f * scale_factor for f in self.fx2)
        self.fy2 = tuple(f * scale_factor for f in self.fy2)
        self.fz2 = tuple(f * scale_factor for f in self.fz2)
    
    def readData(self):
        print("reading data")
        frame_count = self.data.shape[0]//10
        step_size = 600/self.fps
        print(f"This is step_size: {step_size}")


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
            current_row+=step_size

            # the dividing adding by a float is to normalize the data into the positive and then the divide is to normalize the range from 0 to 1 in both directions
            data_x1 = self.data.iloc[start_row, 1].astype('float64')
            data_y1 = self.data.iloc[start_row, 2].astype('float64')
            data_z1 = self.data.iloc[start_row, 3].astype('float64')
            pressure_x1 = (-self.data.iloc[start_row, 5].astype('float64')+0.3)/0.6
            pressure_y1 = (self.data.iloc[start_row, 6].astype('float64')+0.45)/0.9


            # the dividing adding by a float is to normalize the data into the positive and then the divide is to normalize the range from 0 to 1 in both directions
            data_x2 = self.data.iloc[start_row, 10].astype('float64')
            data_y2 = self.data.iloc[start_row, 11].astype('float64')
            data_z2 = self.data.iloc[start_row, 12].astype('float64')
            pressure_x2 = (-self.data.iloc[start_row, 14].astype('float64')+0.3)/0.6
            pressure_y2 = (self.data.iloc[start_row, 15].astype('float64')+0.45)/0.9

            # we are not using the mean
            # fx1.append(data_x1.mean())
            # I only take the start row

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

    def drawArrows(self, frame, xf1, xf2, yf1, yf2, px1, px2, py1, py2):

        # already convert pressure to percentage when reading data, no need to do it here
        # the rect_to_trapezoid translates the normalized force data to the trapazoid that we see of the forceplate surface in the video for force plate 1
        start_point_1 = rect_to_trapezoid(px1, py1, 1, 1,
                                          [self.corners[0], self.corners[1], self.corners[2], self.corners[3]])

        # the rect_to_trapezoid translates the normalized force data to the trapazoid that we see of the forceplate surface in the video for force plate 2
        start_point_2 = rect_to_trapezoid(px2, py2, 1, 1,
                                          [self.corners[4], self.corners[5], self.corners[6], self.corners[7]])
        # print(f"Startpoint1: {start_point_1}, Startpoint2:{start_point_2}")

        end_point_1 = (int(start_point_1[0] + xf1), int(start_point_1[1] - yf1))
        end_point_2 = (int(start_point_2[0] + xf2), int(start_point_2[1] - yf2))

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

            px1 = self.py1[int(frame_number)]
            py1 = self.px1[int(frame_number)]
            px2 = self.py2[int(frame_number)]
            py2 = self.px2[int(frame_number)]

            self.drawArrows(frame, fx1, fx2, fy1, fy2, px1, px2, py1, py2)
            cv2.imshow("window", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_number += 1
            out.write(frame)

        cap.release()
        out.release()
        print(f"Finished processing video; Total Frames: {frame_number}")
# short view need more test
    def ShortVectorOverlay(self,outputName):
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
            fx1 = -self.fx1[int(frame_number)]
            fx2 = -self.fx2[int(frame_number)]
            fy1 = self.fz1[int(frame_number)]
            fy2 = self.fz2[int(frame_number)]
            px1 = self.px1[int(frame_number)]
            px2 = self.px2[int(frame_number)]
            py1 = 1-self.py1[int(frame_number)]
            py2 = 1-self.py2[int(frame_number)]

            self.drawArrows(frame, fx1, fx2, fy1, fy2, px1, px2, py1, py2)
            cv2.imshow("window", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_number += 1
            out.write(frame)

        cap.release()
        out.release()
        print(f"Finished processing video; Total Frames: {frame_number}")





"""
use "\\" if you are in windows
use "/" if you are in ios or windows
"""
folder = "data\\mbm"

# these are the file paths
long_view, short_view, top_view, forcedata = find_files(folder)


# verify file path
print(f"This is top view path: {top_view}\n"
      f"This is long view path: {long_view}\n"
      f"This is short view path: {short_view}\n")
v = VectorOverlay(top_view, long_view, short_view, forcedata)

"""
side view / long view
"""
if long_view != None:
    output_name = outputname(long_view)
    print(f"output file name: {output_name}")
    v.LongVectorOverlay(output_name)

"""
top view
"""
if top_view != None:
    outputName = outputname(top_view)
    print(f"output file name: {outputName}")
    v.TopVectorOverlay(outputName)

"""
front view / short view
"""
if short_view != None:
    outputName = outputname(short_view)
    print(f"output file name: {outputName}")
    v.ShortVectorOverlay(outputName)
