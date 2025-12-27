import cv2
import pandas as pd

class VideoState:
    def __init__(self):
        self.path = None
        self.original_path = None  # Original video before trimming
        self.trimmed_path = None   # Trimmed video for processing
        self.cam = None
        self.vector_cam = None
        self.total_frames = 0
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
        self.led_signal = pd.DataFrame()
        self.view = ""
        
        # Trim boundaries (in ORIGINAL video frame numbers)
        self.trim_start_frame = 0
        self.trim_end_frame = 0
        self.has_trim_boundaries = False
        
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
        
    def get_processing_range(self):
        """
        Get the frame range that should be processed (based on trim boundaries).
        
        Returns:
            tuple: (start_frame, end_frame) - inclusive range
        """
        if self.has_trim_boundaries:
            return (self.trim_start_frame, self.trim_end_frame)
        else:
            return (0, self.total_frames - 1)
    
    def get_processing_frame_count(self):
        """
        Get the number of frames that should be processed.
        
        Returns:
            int: Number of frames to process
        """
        start, end = self.get_processing_range()
        return end - start + 1

# import cv2
# import pandas as pd

# class VideoState:
#     def __init__(self):
#         self.path = None
#         self.cam = None
#         self.vector_cam = None
#         self.total_frames = 0
#         self.frame_width = 0
#         self.frame_height = 0
#         self.fps = 0
#         self.led_signal = pd.DataFrame() # Store square wave signal from a csv file
#         # Create variable to store the template matching stuff
#         self.view = ""

#     def load(self, path):
#         self.path = path
#         self.cam = cv2.VideoCapture(path)
#         self.vector_cam = cv2.VideoCapture(path)  # Secondary video stream
        
#         if not self.cam.isOpened():
#             raise IOError(f"Could not open video at {path}")

#         self.total_frames = int(self.cam.get(cv2.CAP_PROP_FRAME_COUNT))
#         self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
#         self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         self.fps = int(self.cam.get(cv2.CAP_PROP_FPS))
