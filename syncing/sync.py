from forceplate_detect import ForcePlateDetect
from moviepy.editor import *


fp_detect = ForcePlateDetect("data/5.5min_120Hz_SSRun_Fa19_OL_skele.mp4") # 60.0 fps
frameNum, fps = fp_detect.detect((768, 790), (430, 25), False)

lowerTime = (1/fps) * (frameNum-10)
upperTime = (1/fps) * (frameNum+10)

clip = VideoFileClip("data/5.5min_120Hz_SSRun_Fa19_OL_skele.mp4").subclip(lowerTime, upperTime)
CompositeVideoClip([clip]).write_videofile("saved.mp4", fps=fps)

# runs the main detection loop to find the first frame when the force plate is triggered
