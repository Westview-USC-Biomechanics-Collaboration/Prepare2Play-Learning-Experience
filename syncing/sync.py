from forceplate_detect import ForcePlateDetect
from plt_graph import Graph
from moviepy.editor import *
import pandas as pd


# TODO: REFACTOR


class VideoSync():
    def __init__(self, videoPath, graphPath):
        self.videoPath = videoPath
        self.graphPath = graphPath

    def syncSave(self):
        fp_detect = ForcePlateDetect(self.videoPath)  # 60.0 fps, frame 240
        movementFrameNum, fps = fp_detect.detect((768, 790), (430, 25), False)
        # runs the main detection loop to find the first frame when the force plate is triggered

        movementClip = VideoFileClip(self.videoPath)

        df = pd.read_csv(self.graphPath)

        forcey_subset = {"data": df.iloc[18:10000, 2].astype(float).tolist(), "name": "ForceY", "min": -15, "max": 15}
        timex_subset = {"data": df.iloc[18:10000, 0].astype(float).tolist(), "name": "Time", "min": 0, "max": 3}

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


sync = VideoSync("data/5.5min_120Hz_SSRun_Fa19_OL_skele.mp4", "data/SM_SbS_02_Raw_Data - SM_SoftballSwing_Trial2_Raw_Data.csv")
sync.syncSave()
print("Video Generated!")