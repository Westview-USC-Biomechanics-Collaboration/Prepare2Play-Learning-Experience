import cv2
from PIL import Image, ImageTk

class FrameConverter:
    def __init__(self):
        pass
    # def cvToPillow(self, camera:cv2.VideoCapture, width=400, height=300):
    #     ret, frame = camera.read()  # the `frame` object is now the frame we want

    #     if ret:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    #         frame = Image.fromarray(frame).resize((width, height), resample=Image.BICUBIC) # Resize the frame to 400 * 300
    #         photoImage = ImageTk.PhotoImage(frame)   # ---> update the image object base on current frame.
    #         return photoImage
    def cvToPillow(self, camera, width=None, height=None, frame_number=None):
        if frame_number is not None:
            camera.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = camera.read()
        if not ret:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if width and height:
            frame = cv2.resize(frame, (width, height))

        return ImageTk.PhotoImage(Image.fromarray(frame))


    def cvToPillowFromFrame(self, frame:cv2.Mat, width=400, height=300):
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            frame = Image.fromarray(frame).resize((width, height), resample=Image.BICUBIC)
            photoImage = ImageTk.PhotoImage(frame)   # ---> update the image object base on current frame.
            return photoImage