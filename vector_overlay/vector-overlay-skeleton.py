# Import libraries
import cv2
import cv2 as cv
import pandas as pd
import math
from corner_detect import FindCorners
from corner_detect import Views
from vector_overlay_top import VectorOverlay as Topview
from test_corners import select_points
import numpy as np
import os
# Initialization
def outputname(path):
    if "top" in path:
        output_name = "outputs\\" + path[-23:-4] + "_vector_overlay.mp4"
    else:
        output_name = "outputs\\" + path[-20:-4] + "_vector_overlay.mp4"

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

    mp4_files.sort(key=lambda x: len(os.path.basename(x)))

    # top, side, data
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
    
    return new_x, new_y

class VectorOverlay:

    def __init__(self, top_view_path, side_view_path, data_path,smooth = False):
        self.top_view_path = top_view_path
        self.side_view_path = side_view_path
        self.data_path = data_path
        df = pd.read_excel(self.data_path, skiprows= 18)
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

        self.A_1 = ()  # ([Ax], [Ay])
        self.A_2 = ()  # ([Ax], [Ay])

        self.force_1 = ()  # ([Y], [Z])
        self.force_2 = ()  # ([Y], [Z])

        self.corners = FindCorners(self.side_view_path).find(Views.Side)  # [482,976] [959,977]
        # self.corners = [482,976],[959,977],[966,976]
        self.manual = False
        self.smooth = smooth

    def check_corner(self):
        # print("Checking the corners")
        if self.corners == []:
            print("Need human force")
            self.manual = True
            self.corners = select_points(video_path=self.side_view_path)

    def setFrameData(self):
        print(f"Opening video: {self.side_view_path}")
        cap = cv.VideoCapture(self.side_view_path)

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

    def readData(self):
        frame_width = self.frame_height
        frame_height = self.frame_width
        fps = self.fps
        frame_count = self.frame_count

        df = pd.read_excel(self.data_path, skiprows= 18)
        self.data = df
        print(self.data.iloc[:4,:])

        rows = self.data.shape[0]
        step_size = rows/frame_count

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
            current_row+=step_size
        
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


        
        numeric_df = df.select_dtypes(include=[np.number])

        if self.smooth:
        # Apply rolling average to smooth data
            window_size = 5  # You can adjust this value for smoother results
            df_smoothed = numeric_df.rolling(window=window_size, min_periods=1).mean()
            if df_smoothed.shape[1] > 15 and len(df_smoothed) > 18:
                self.force_1 = (
                df_smoothed.iloc[18:, 2].astype(float).tolist(), df_smoothed.iloc[18:, 3].astype(float).tolist())
                self.force_2 = (
                df_smoothed.iloc[18:, 11].astype(float).tolist(), df_smoothed.iloc[18:, 12].astype(float).tolist())
                self.A_1 = (
                df_smoothed.iloc[18:, 5].astype(float).tolist(), df_smoothed.iloc[18:, 6].astype(float).tolist())
                self.A_2 = (
                df_smoothed.iloc[18:, 14].astype(float).tolist(), df_smoothed.iloc[18:, 15].astype(float).tolist())
            else:
                print("Error: DataFrame does not have the required number of rows or columns.")
        else:
            self.force_1 = (df.iloc[18:, 2].astype(float).tolist(), df.iloc[18:, 3].astype(float).tolist())
            self.force_2 = (df.iloc[18:, 11].astype(float).tolist(), df.iloc[18:, 12].astype(float).tolist())

            self.A_1 = (df.iloc[18:, 5].astype(float).tolist(), df.iloc[18:, 6].astype(float).tolist())
            self.A_2 = (df.iloc[18:, 14].astype(float).tolist(), df.iloc[18:, 15].astype(float).tolist())
        print(f"Data read successfully from {self.data_path}")
        print(f"Number of frames of force data: {len(self.force_1[0])}")

    def drawArrows(self, frameNum, frame):
        z_force_1 = self.force_1[1][frameNum]
        y_force_1 = self.force_1[0][frameNum]

        z_force_2 = self.force_2[1][frameNum]
        y_force_2 = self.force_2[0][frameNum]
        self.check_corner()
        # print(f"corners: {self.corners}")
        if self.manual == False:
            force_plate_pixels = self.corners[1][0] - self.corners[0][0]
            force_plate_meters = 0.9
            pixelOffset_1 = (force_plate_pixels / force_plate_meters) * self.A_1[1][frameNum] + 0.45 * (
                    force_plate_pixels / force_plate_meters)  # Ay_1
            start_point_1 = (self.corners[0][0] + round(pixelOffset_1),
                             self.corners[0][1])  # a negative Ay val means moving to the right
            end_point_1 = (start_point_1[0] - int(y_force_1), (start_point_1[1] - int(z_force_1)))
            pixelOffset_2 = (force_plate_pixels / force_plate_meters) * self.A_2[1][frameNum] + 0.45 * (
                    force_plate_pixels / force_plate_meters)  # Ay_2
            start_point_2 = (self.corners[2][0] + round(pixelOffset_2), self.corners[2][1])
            end_point_2 = (start_point_2[0] - int(y_force_2), (start_point_2[1] - int(z_force_2)))
        else:
            # print(f"This is corner list: {self.corners}")
            force_plate_pixels = self.corners[6][0] - self.corners[7][0]
            force_plate_meters = 0.9
            pixelOffset_1 = (force_plate_pixels / force_plate_meters) * self.A_1[1][frameNum] + 0.45 * (
                    force_plate_pixels / force_plate_meters)  # Ay_1
            start_point_1 = (self.corners[7][0] + round(pixelOffset_1),
                             self.corners[7][1])  # a negative Ay val means moving to the right

            end_point_1 = (start_point_1[0] - int(y_force_1), (start_point_1[1] - int(z_force_1)))

            pixelOffset_2 = (force_plate_pixels / force_plate_meters) * self.A_2[1][frameNum] + 0.45 * (
                    force_plate_pixels / force_plate_meters)  # Ay_2
            start_point_2 = (self.corners[5][0] + round(pixelOffset_2), self.corners[5][1])

            end_point_2 = (start_point_2[0] - int(y_force_2), (start_point_2[1] - int(z_force_2)))
        cv.arrowedLine(frame, start_point_1, end_point_1, (0, 255, 0), 2)

        cv.arrowedLine(frame, start_point_2, end_point_2, (255, 0, 0), 2)

    def createVectorOverlay(self, outputName):
        self.setFrameData()
        self.readData()

        if self.frame_width is None or self.frame_height is None:
            print("Error: Frame data not set.")
            return

        if self.force_1 is None or self.force_2 is None:
            print("Error: Data not set.")
            return

        out = cv.VideoWriter(outputName, cv.VideoWriter_fourcc(*'mp4v'), self.fps,
                             (self.frame_width, self.frame_height))

        cap = cv.VideoCapture(self.side_view_path)
        frame_number = 1
        forceDataLength = len(self.force_1[0])
        speedMult = math.floor(forceDataLength / self.frame_count)

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

    def getCorners(self):
        cap = cv.VideoCapture(self.side_view_path)

        # tl_1, tr_1, tl_2, tr_2, bl_1, br_1, bl_2, br_2
        coords = [(570, 900), (965, 900), (975, 900), (1370, 900), (485, 978), (958, 978), (965, 978), (1445, 978)]

        if not cap.isOpened():
            print("Error: Could not open video. ")
            return

        while True:
            ret, frame = cap.read()
            # forceplate1 = left, forceplate2 = left
            for coord in coords:
                cv.circle(frame, coord, 3, (0, 255, 0), 3)

            if not ret:
                print("Can't find frame")
                break

            cv.imshow("frame", frame)

            if cv.waitKey(1) == ord("q"):  # gets the unicode value for q
                break
        cap.release()
        cv.destroyAllWindows()
        return coords

folder = "data\\Formated_files"

#these are the file paths
top_view, side_view, forcedata = get_files_from_folder(folder)

smoothed_data = False



v = VectorOverlay(top_view, side_view, forcedata,smoothed_data)

# side view
output_name = outputname(side_view)
print(f"output file name: {output_name}")
v.createVectorOverlay(output_name)

# top view
# outputName = outputname(top_view)
# print(f"output file name: {outputName}")
# points_loc = select_points(v.top_view_path)
# Topview(top_view, v.data, outputName, points_loc, smoothed_data)