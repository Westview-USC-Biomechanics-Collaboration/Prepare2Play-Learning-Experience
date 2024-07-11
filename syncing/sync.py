from forceplate_detect import ForcePlateDetect
from plt_graph import Graph
from moviepy.editor import *
import pandas as pd


class VideoSync():
    def __init__(self, videoPath, csvPath):
        self.videoPath = videoPath
        self.csvPath = csvPath

    def syncSave(self): # camera is 240 fps
        fp_detect = ForcePlateDetect(self.videoPath)
        movementFrameNum, fps = fp_detect.detect((550, 800), (247, 97), False)
        # runs the main detection loop to find the first frame when the force plate is triggered

        movementClip = VideoFileClip(self.videoPath)

        df = pd.read_csv(self.csvPath)

        forcey_subset = {"data": df.iloc[18:, 2].astype(float).tolist(), "name": "ForceY"}
        timex_subset = {"data": df.iloc[18:, 0].astype(float).tolist(), "name": "Time"}

        saveAs = "syncing/results/graph.mp4"
        g = Graph(timex_subset, forcey_subset)
        g.animate_graph(True, saveAs)
        graphFrameNum = g.getForcePlateTime()

        graphClip = VideoFileClip(saveAs)

        frameHoldLength = movementFrameNum - graphFrameNum  # frames of filler before the graph starts

        blankFrame = ColorClip(graphClip.size, (0, 0, 0), duration=(1 / fps) * frameHoldLength)

        finalizedGraphClip = concatenate_videoclips([blankFrame, graphClip])
        clips = [movementClip], [finalizedGraphClip]

        final = clips_array(clips)
        final.write_videofile("syncing/results/syncedVideo.mp4", fps=fps)


videoPath = "data/NS_SPU_01.mov"
csvPath = "data/NS_SPU_01_Raw_Data - NS_SurfPopUp_Trial1_Raw_Data.csv"

sync = VideoSync(videoPath, csvPath)
sync.syncSave()
print("Video Generated!")
