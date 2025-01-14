
import cv2


# @dataclass
class Videostate:
    path: str = None
    cam: cv2.VideoCapture = None
    total_frame: int = 100


print(Videostate.total_frame)
