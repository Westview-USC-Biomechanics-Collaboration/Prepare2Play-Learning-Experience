from forceplate_detect import ForcePlateDetect
from moviepy.editor import *

fp_detect = ForcePlateDetect("data/5.5min_120Hz_SSRun_Fa19_OL_skele.mp4")  # 60.0 fps, frame 240
movementFrameNum, fps = fp_detect.detect((768, 790), (430, 25), False)
# runs the main detection loop to find the first frame when the force plate is triggered

lowerTime = (1 / fps) * (movementFrameNum - 10)
upperTime = (1 / fps) * (movementFrameNum + 10)
# .subclip(lowerTime, upperTime)
movementClip = VideoFileClip("data/5.5min_120Hz_SSRun_Fa19_OL_skele.mp4")

graphClip = VideoFileClip("data/data.mov")
# 1.33 s @ 60 fps, detection at frame 80 <-- this data would be received from the graphs (this is temporary)
graphFrameNum = 80
frameHoldLength = movementFrameNum - graphFrameNum  # frames of filler before the graph starts

blankFrame = ColorClip(graphClip.size, (0, 0, 0), duration=(1/fps) * frameHoldLength)

finalizedGraphClip = concatenate_videoclips([blankFrame, graphClip])

clips = [movementClip], [finalizedGraphClip]

final = clips_array(clips)
final.write_videofile("saved.mp4", fps=fps)
