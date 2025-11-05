import cv2
import pandas as pd

class VideoState:
    def __init__(self):
        self.path = None
        self.cam = None
        self.vector_cam = None
        self.total_frames = 0
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
        self.led_signal = pd.DataFrame() # Store square wave signal from a csv file
        # Create variable to store the template matching stuff
        self.view = ""

    def load(self, path):
        self.path = path
        self.cam = cv2.VideoCapture(path)
        self.vector_cam = cv2.VideoCapture(path)  # Secondary video stream

        if not self.cam.isOpened():
            raise IOError(f"Could not open video at {path}")

        self.total_frames = int(self.cam.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cam.get(cv2.CAP_PROP_FPS))
