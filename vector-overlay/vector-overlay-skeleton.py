# Import libraries
import cv2 as cv
import pandas as pd
import openpyxl

# initialization
top_view = "data/gis_lr_CC_top_vid03.mp4"
side_view = "data/gis_lr_CC_vid03.mov"
forcedata_path = "../data/gis_lr_CC_for03_Raw_Data.xlsx"
output_name = top_view[5:-4] + "_vector_overlay"


class VectorOverlay:

    def __init__(self, top_view_path, side_view_path, data_path):
        self.top_view_path = top_view_path
        self.side_view_path = side_view_path
        self.data_path = data_path

        self.frame_width, self.frame_height, self.fps, self.frame_count = None, None, None, None

    def setFrameData(self):
        cap = cv.VideoCapture(self.side_view_path)

        if not cap.isOpened():
            print("Error: Could not open video. ")
            return

        self.frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv.CAP_PROP_FPS))
        self.frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        cap.release()

    def VectorOverlay(self, outputName):
        self.setFrameData()

        # Define the codec and create VideoWriter object to save the annotated video
        out = cv.VideoWriter(outputName, cv.VideoWriter_fourcc(*'mp4v'), self.fps,
                             (self.frame_width, self.frame_height))

        cap = cv.VideoCapture(self.side_view_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't find frame")
                break

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
v.setFrameData()
v.findCorners()
