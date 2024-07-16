# Import libraries
import cv2 as cv
import pandas as pd
import openpyxl

# Initialization
top_view = "data/gis_lr_CC_top_vid03.mp4"
side_view = "data/bcp_lr_CC_vid02.mp4"
forcedata_path = "data\gis_lr_CC_for02_Raw_Data.xlsx"
output_name = top_view[5:-4] + "_vector_overlay.mp4"


class VectorOverlay:

    def __init__(self, top_view_path, side_view_path, data_path):
        self.top_view_path = top_view_path
        self.side_view_path = side_view_path
        self.data_path = data_path

        self.frame_width, self.frame_height, self.fps, self.frame_count = None, None, None, None
        self.x_forces = None
        self.y_forces = None

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
        df = pd.read_excel(self.data_path)
        self.x_forces = df.iloc[18:18193, 1].astype(float).tolist()
        self.y_forces = df.iloc[18:18192, 2].astype(float).tolist()
        print(f"Data read successfully from {self.data_path}")
        print(f"Number of frames of force data: {len(self.x_forces)}")

    def createVectorOverlay(self, outputName):
        self.setFrameData()
        self.readData()

        if self.frame_width is None or self.frame_height is None:
            print("Error: Frame data not set.")
            return

        if self.x_forces is None or self.y_forces is None:
            print("Error: Data not set.")
            return

        out = cv.VideoWriter(outputName, cv.VideoWriter_fourcc(*'mp4v'), self.fps,
                             (self.frame_width, self.frame_height))

        cap = cv.VideoCapture(self.side_view_path)
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"Can't read frame at position {frame_number}")
                break

            if frame_number < len(self.x_forces):
                x_force = self.x_forces[frame_number]
                y_force = self.y_forces[frame_number]

                start_point = (self.frame_width // 2, self.frame_height // 2)

                end_point = (start_point[0] + int(x_force), start_point[1] + int(y_force))

                cv.arrowedLine(frame, start_point, end_point, (0, 255, 0), 2)

            out.write(frame)
            frame_number += 1

        cap.release()
        out.release()
        print("Finished processing video")

    def findCorners(self):
        cap = cv.VideoCapture(self.side_view_path)

        # tl_1, tr_1, tl_2, tr_2, bl_2, br_1, bl_2, br_2
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


v = VectorOverlay(top_view, side_view, forcedata_path)

v.createVectorOverlay(output_name)
