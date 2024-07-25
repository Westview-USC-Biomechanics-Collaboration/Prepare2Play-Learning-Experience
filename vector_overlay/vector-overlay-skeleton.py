# Import libraries
import cv2
import cv2 as cv
import pandas as pd
import math
from vector_overlay_top import VectorOverlay as Topview
from test_corners import select_points
import numpy as np
import os
"""force plate directions(at top view)
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
       1              2 3          4
      /---------------||----------\\
     /               | |            \\
   8/---------------|7 6|-------------\\5
    |---------------|   |--------------|
    
    top view:
    1               2  3                4
     _______________    ________________
    |               |  |                |
    |               |  |                |
    |               |  |                |
    ----------------   ----------------- 
    8               7  6                5
    
    shortview/ front view:
                1        2
                __________
               /          \\
              /            \\
          8  |--------------| 3
           7__________________   4
           /                  \\
          /                    \\
       6 /______________________\\ 5     
         |_______________________|
    
"""


# Initialization
def outputname(path):
    if "top" in path:
        output_name = "outputs/" + path[-23:-4] + "_vector_overlay.mp4"
    else:
        output_name = "outputs/" + path[-20:-4] + "_vector_overlay.mp4"

    return output_name


def get_files_from_folder(folder):
    xlsx_file = None
    mp4_files = []

    # Check if the folder exists
    if not os.path.isdir(folder):
        return None, None, None

    # Iterate through files in the folder
    for file in os.listdir(folder):
        if file.endswith('.xlsx'):
            xlsx_file = os.path.join(folder, file)
        elif file.endswith('.mp4'):
            mp4_files.append(os.path.join(folder, file))

        # Break the loop if we have found all required files
        if xlsx_file and len(mp4_files) == 2:
            break

    # If we didn't find exactly two .mp4 files and one .xlsx file, return None
    if len(mp4_files) != 2 or xlsx_file is None:
        return None, None, None

    mp4_files.sort(key=lambda x: len(os.path.basename(x)), reverse=False)

    # top, long, data
    return mp4_files[1], mp4_files[0], xlsx_file


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


def find_point(startpoint, angle, x_in, y_in):
    """
    start point is the (0,0) of the system
    angle is the angle between the force plate and the horizontal view.
    x_in is the location of the point in pixels
    y_in is the location of the point in pixels
    """
    angle2 = math.atan(y_in / x_in)

    hypotenuse = math.sqrt(x_in ** 2 + y_in ** 2)
    if (x_in > 0 and y_in > 0) or (x_in > 0 and y_in < 0):  # first and fourth quadrant
        angle3 = angle + angle2
    elif (x_in < 0 and y_in < 0) or (y_in > 0 and x_in < 0):  # second and third quadrant
        angle3 = angle + angle2 + math.pi
    else:
        # print(f"\nThis is startpoint: {startpoint}")
        endpoint = [0] * len(startpoint)
        # print(f"\nThis is x in , y in: {x_in,y_in}")
        for i in range(len(startpoint)):
            if x_in != 0:
                endpoint.append(startpoint[i] + x_in)
            elif y_in != 0:
                endpoint.append(startpoint[i] + y_in)
            else:
                endpoint[i] = startpoint[i]

        # print(f"\nThe function find_point is going to return: {endpoint}")
        return endpoint


    deltax = hypotenuse * math.cos(angle3)
    deltay = hypotenuse * math.sin(angle3)
    endpoint = [startpoint[0] + deltax, startpoint[1] - deltay]
    return endpoint


def convert_floats_to_integers(lst):
    """
    This function takes a list and converts all float elements to integers.

    Parameters:
    lst (list): The input list containing elements of various types.

    Returns:
    list: The modified list with all float elements converted to integers.
    """
    return [int(x) if isinstance(x, float) else x for x in lst]


class VectorOverlay:

    def __init__(self, top_view_path, long_view_path, short_view_path, data_path):
        self.top_view_path = top_view_path
        self.long_view_path = long_view_path
        self.short_view_path = short_view_path

        self.data_path = data_path
        df = pd.read_excel(self.data_path, skiprows=18)
        self.data = df

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

    def check_corner(self, path):
        self.corners = select_points(video_path=path)

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

    # You can adjust speed Mult in the readData function!!!
    def readData(self):
        print("reading data")
        frame_count = self.frame_count
        rows = self.data.shape[0]
        step_size = rows / frame_count
        # Adjust speed mult here:
        step_size = 10
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

            data_x1 = self.data.iloc[start_row:end_row, 1].astype('float64')
            data_y1 = self.data.iloc[start_row:end_row, 2].astype('float64')
            data_z1 = self.data.iloc[start_row:end_row, 3].astype('float64')
            pressure_x1 = self.data.iloc[start_row:end_row, 5].astype('float64')
            pressure_y1 = self.data.iloc[start_row:end_row, 6].astype('float64')

            data_x2 = self.data.iloc[start_row:end_row, 10].astype('float64')
            data_y2 = self.data.iloc[start_row:end_row, 11].astype('float64')
            data_z2 = self.data.iloc[start_row:end_row, 12].astype('float64')
            pressure_x2 = self.data.iloc[start_row:end_row, 14].astype('float64')
            pressure_y2 = self.data.iloc[start_row:end_row, 15].astype('float64')

            fx1.append(data_x1.mean())
            fy1.append(data_y1.mean())
            fz1.append(data_z1.mean())
            px1.append(pressure_x1.mean())
            py1.append(pressure_y1.mean())

            fx2.append(data_x2.mean())
            fy2.append(data_y2.mean())
            fz2.append(data_z2.mean())
            px2.append(pressure_x2.mean())
            py2.append(pressure_y2.mean())


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

    def drawArrows(self, frameNum, frame):
        z_force_1 = self.fz1[frameNum]
        y_force_1 = self.fy1[frameNum]

        z_force_2 = self.fz2[frameNum]
        y_force_2 = self.fy2[frameNum]
        
        # print(f"corners: {self.corners}")

        start_point_1 = rect_to_trapezoid(self.py1[frameNum] + 0.45, self.px1[frameNum] + 0.3, 0.9, 0.6,
                                          [self.corners[0], self.corners[1], self.corners[6], self.corners[7]])
        start_point_2 = rect_to_trapezoid(self.py2[frameNum] + 0.45, self.px1[frameNum] + 0.3, 0.9, 0.6,
                                          [self.corners[2], self.corners[3], self.corners[4], self.corners[5]])

        end_point_1 = (int(start_point_1[0] - y_force_1), int(start_point_1[1] - z_force_1))
        end_point_2 = (int(start_point_2[0] - y_force_2), int(start_point_2[1] - z_force_2))
        # print(start_point_1)
        # print(end_point_1)
        #
        # print(start_point_2)
        # print(end_point_2)
        cv.arrowedLine(frame, start_point_1, end_point_1, (0, 255, 0), 2)

        cv.arrowedLine(frame, start_point_2, end_point_2, (255, 0, 0), 2)

        # Draw red dots for centers
        cv.circle(frame, self.corners[0], 5, (0, 0, 255), -1)  # Red dot at start_point_1
        cv.circle(frame, self.corners[1], 5, (0, 0, 255), -1)  # Red dot at end_point_1
        cv.circle(frame, self.corners[2], 5, (0, 0, 255), -1)  # Red dot at start_point_2
        cv.circle(frame, self.corners[3], 5, (0, 0, 255), -1)  # Red dot at end_point_2
        cv.circle(frame, self.corners[4], 5, (0, 0, 255), -1)  # Red dot at start_point_1
        cv.circle(frame, self.corners[5], 5, (0, 0, 255), -1)  # Red dot at end_point_1
        cv.circle(frame, self.corners[6], 5, (0, 0, 255), -1)  # Red dot at start_point_2
        cv.circle(frame, self.corners[7], 5, (0, 0, 255), -1)  # Red dot at end_point_2

    def LongVectorOverlay(self, outputName):
        self.setFrameData(path=self.long_view_path)
        self.readData()

        if self.frame_width is None or self.frame_height is None:
            print("Error: Frame data not set.")
            return

        if self.fz1 is None or self.fz2 is None:
            print("Error: Data not set.")
            return

        out = cv.VideoWriter(outputName, cv.VideoWriter_fourcc(*'mp4v'), self.fps,
                             (self.frame_width, self.frame_height))

        cap = cv.VideoCapture(self.long_view_path)
        frame_number = 1
        forceDataLength = len(self.fz1)
        speedMult = math.floor(forceDataLength / self.frame_count)
        self.check_corner(self.long_view_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # if this calls when the frame_number is equal to the total frame count then the stream has just ended
                print(f"Can't read frame at position {frame_number}")
                break

            self.drawArrows(frame_number * speedMult, frame)
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

        if self.frame_width is None or self.frame_height is None:
            print("Error: Frame data not set.")
            return

        if self.fz1 is None or self.fz2 is None:
            print("Error: Data not set.")
            return

        out = cv.VideoWriter(outputName, cv.VideoWriter_fourcc(*'mp4v'), self.fps,
                             (self.frame_width, self.frame_height))

        cap = cv.VideoCapture(self.top_view_path)
        frame_number = 1
        forceDataLength = len(self.fz1)

        self.check_corner(self.top_view_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # if this calls when the frame_number is equal to the total frame count then the stream has just ended
                print(f"Can't read frame at position {frame_number}")
                break

            self.drawTopArrows(frame_number, frame)
            cv2.imshow("window", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_number += 1
            out.write(frame)

        cap.release()
        out.release()
        print(f"Finished processing video; Total Frames: {frame_number}")

    def drawTopArrows(self, frameNum, frame):
        # convert ax, ay to pixel on plate system
        y_diff = (self.corners[5][1] - self.corners[7][1])
        x_diff = (self.corners[5][0] - self.corners[7][0])
        meter_pixel_ratio = math.sqrt(y_diff ** 2 + x_diff ** 2) / 0.9  # This tell us how many pixel represent 1 meter
        angle_of_forceplate = math.atan(
            y_diff / x_diff)  # This is the angle between the force plate and the horizontal view
        # print(f"frameNum = {frameNum}")
        x_force_1 = self.fx1[frameNum]
        y_force_1 = self.fy1[frameNum]
        ax_1 = self.px1[frameNum] * meter_pixel_ratio
        ay_1 = self.py1[frameNum] * meter_pixel_ratio

        x_force_2 = self.fx2[frameNum]
        y_force_2 = self.fy2[frameNum]
        ax_2 = self.px2[frameNum] * meter_pixel_ratio
        ay_2 = self.py2[frameNum] * meter_pixel_ratio
        #
        # print(f"This is px2: {self.px2}")
        # print(f"corners: {self.corners}")

        # find the center of the forceplate
        center1 = find_point(self.corners[7], angle_of_forceplate, 0.45 * meter_pixel_ratio, 0.3 * meter_pixel_ratio)
        center2 = find_point(self.corners[5], angle_of_forceplate, 0.45 * meter_pixel_ratio, 0.3 * meter_pixel_ratio)
        # print("\n"+ "centers:")
        # print(center1)
        # print(center2)

        start_point_1 = find_point(center1, angle_of_forceplate, ay_1, ax_1)
        start_point_2 = find_point(center2, angle_of_forceplate, ay_2, ax_2)

        end_point_1 = find_point(start_point_1, angle_of_forceplate, y_force_1 * -10, x_force_1 * -10)
        end_point_2 = find_point(start_point_2, angle_of_forceplate, y_force_2 * -10, x_force_2 * -10)

        # convert list float element to integer
        center1 = convert_floats_to_integers(center1)
        center2 = convert_floats_to_integers(center2)
        start_point_1 = convert_floats_to_integers(start_point_1)
        start_point_2 = convert_floats_to_integers(start_point_2)

        end_point_1 = convert_floats_to_integers(end_point_1)
        end_point_2 = convert_floats_to_integers(end_point_2)

        # print("\n"+"force plate 1:")
        # print(start_point_1)
        # print(end_point_1)
        #
        # print("\n"+"force plate 2:")
        # print(start_point_2)
        # print(end_point_2)

        #draw plate center
        cv.circle(frame, center1, 5, (255, 0, 0), -1)  #draw dot at center 1
        cv.circle(frame, center2, 5, (255, 0, 0), -1)  # draw dot at center 2

        # drwa green dot at start points
        cv.circle(frame, start_point_1, 5, (0, 255, 0), -1)  # draw dot at startpoint 1
        cv.circle(frame, start_point_2, 5, (0, 255, 0), -1)  # draw dot at startpoint 2

        # draw arrowline
        cv.arrowedLine(frame, start_point_1, end_point_1, (0, 255, 0), 2)
        cv.arrowedLine(frame, start_point_2, end_point_2, (255, 0, 0), 2)

        # Draw red dots for centers
        cv.circle(frame, self.corners[0], 5, (0, 0, 255), -1)  # Red dot at start_point_1
        cv.circle(frame, self.corners[1], 5, (0, 0, 255), -1)  # Red dot at end_point_1
        cv.circle(frame, self.corners[2], 5, (0, 0, 255), -1)  # Red dot at start_point_2
        cv.circle(frame, self.corners[3], 5, (0, 0, 255), -1)  # Red dot at end_point_2
        cv.circle(frame, self.corners[4], 5, (0, 0, 255), -1)  # Red dot at start_point_1
        cv.circle(frame, self.corners[5], 5, (0, 0, 255), -1)  # Red dot at end_point_1
        cv.circle(frame, self.corners[6], 5, (0, 0, 255), -1)  # Red dot at start_point_2
        cv.circle(frame, self.corners[7], 5, (0, 0, 255), -1)  # Red dot at end_point_2


folder = "data/Jiya"

# these are the file paths
top_view, long_view, forcedata = get_files_from_folder(folder)
sideViewPath = None

# verify file path
print(f"This is top view path: {top_view}\n"
      f"This is long view path: {long_view}")
v = VectorOverlay(top_view, long_view, sideViewPath, forcedata)

# long view
# output_name = outputname(long_view)
# print(f"output file name: {output_name}")
# v.LongVectorOverlay(output_name)

# top view
outputName = outputname(top_view)
print(f"output file name: {outputName}")
v.TopVectorOverlay(outputName)

# short view
# outputName = outputName(sideViewPath)
# print(f"output file name: {outputName}")
