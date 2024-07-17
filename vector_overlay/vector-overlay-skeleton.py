# Import libraries
import cv2 as cv
import pandas as pd
import math
from corner_detect import FindCorners
from corner_detect import Views

# Initialization
top_view = "data/gis_lr_CC_top_vid03.mp4"
side_view = "data/Trimmed of spu_lr_NS_for01_Raw_Video.mp4"
forcedata_path = "data/Trimmed of spu_lr_NS_for01_Raw_Data_new - spu_lr_NS_for01_Raw_Data_new.csv"
output_name = top_view[5:-4] + "_vector_overlay.mp4"


class VectorOverlay:

    def __init__(self, top_view_path, side_view_path, data_path):
        self.top_view_path = top_view_path
        self.side_view_path = side_view_path
        self.data_path = data_path

        self.frame_width, self.frame_height, self.fps, self.frame_count = None, None, None, None
        self.A_1 = ()  # ([Ax], [Ay])
        self.A_2 = ()  # ([Ax], [Ay])

        self.force_1 = ()  # ([Y], [Z])
        self.force_2 = ()  # ([Y], [Z])
        self.corners = FindCorners(self.side_view_path).find(Views.Side)

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
        df = pd.read_csv(self.data_path)
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

        force_plate_pixels = self.corners[1][0] - self.corners[0][0]
        # the x difference in the first two corners on the first forceplate
        force_plate_meters = 1

        pixelOffset_1 = (force_plate_pixels / force_plate_meters) * -self.A_1[1][frameNum]  # Ay_1
        start_point_1 = (self.corners[0][0] + round(pixelOffset_1), self.corners[0][1]) # a negative Ay val means moving to the right

        end_point_1 = (start_point_1[0] + int(y_force_1), (start_point_1[1] - int(z_force_1)))

        pixelOffset_2 = (force_plate_pixels / force_plate_meters) * -self.A_2[1][frameNum]  # Ay_2
        start_point_2 = (self.corners[2][0] + round(pixelOffset_2), self.corners[2][1])

        end_point_2 = (start_point_2[0] + int(y_force_2), (start_point_2[1] - int(z_force_2)))

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


v = VectorOverlay(top_view, side_view, forcedata_path)
v.createVectorOverlay(output_name)
