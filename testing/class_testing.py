import pandas as pd
import cv2
from tkinter import filedialog

#@dataclass
class Video:
    path: str = None,
    cam : cv2.VideoCapture = None,
    vector_cam: cv2.VideoCapture = None,
    total_frames: int = None,
    frame_width: int = None,
    frame_height: int = None,
    fps: int = None,

#@dataclass
class Force:
    path: str = None,
    data: pd.array = None,
    rows: int = None,

#file_path = filedialog.askopenfilename(title="Select Force Data File",filetypes=[("Excel or CSV Files", "*.xlsx *.xls *.csv")])
#Force.data = pd.read_excel(file_path)
Video.path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")])

Video.cam = cv2.VideoCapture(Video.path)
print(Video.cam.isOpened)
