import tkinter as tk
import cv2
import threading
from vector_overlay.vectoroverlay_GUI import VectorOverlay

def vectorOverlayCallback(self):
    def threadTarget():
        process(self)
        self.state.vector_overlay_enabled = True
    vectorOverlayThread = threading.Thread(target=threadTarget,daemon=True)
    vectorOverlayThread.start()

def process(self):
    temp_video = "vector_overlay_temp.mp4"
    self.Video.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
    v = VectorOverlay(data=self.Force.data,video=self.Video.cam)
    
    if(self.selected_view.get()=="Long View"):
        v.LongVectorOverlay(outputName=temp_video)
    elif(self.selected_view.get()=="Short View"):
        v.ShortVectorOverlay(outputName=temp_video)
    elif(self.selected_view.get()=="Top View"):
        v.TopVectorOverlay(outputName=temp_video)

    self.Video.vector_cam = cv2.VideoCapture(temp_video)
    self.Video.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

    """
    display 
    """
    if self.state.loc>=self.state.video_align:
        self.Video.vector_cam.set(cv2.CAP_PROP_POS_FRAMES, self.state.loc - self.state.video_align)
        self.photo_image3 = self.frameConverter.cvToPillow(camera=self.Video.vector_cam)
    else:
        self.Video.cam.set(cv2.CAP_PROP_POS_FRAMES, self.state.loc)
        self.photo_image3 = self.frameConverter.cvToPillow(camera=self.Video.cam)