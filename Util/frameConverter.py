import cv2
from PIL import Image, ImageTk

class FrameConverter:
    def __init__(self):
        pass
    def cvToPillow(self, camera:cv2.VideoCapture, width=400, height=300):
        ret, frame = camera.read()  # the `frame` object is now the frame we want

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            frame = Image.fromarray(frame).resize((width, height), resample=Image.BICUBIC) # Resize the frame to 400 * 300
            photoImage = ImageTk.PhotoImage(frame)   # ---> update the image object base on current frame.
            return photoImage