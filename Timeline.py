import cv2
class timeline():
    def __init__(self,rows,frames,force_label=0,video_label=0):
        self.rows = rows
        self.frames = frames

        self.force_label = force_label
        self.video_label = video_label


    def update_force_label(self,num):
        self.force_label = num

    def update_video_label(self,num):
        self.video_label = num

    def create_rect(self):
        

