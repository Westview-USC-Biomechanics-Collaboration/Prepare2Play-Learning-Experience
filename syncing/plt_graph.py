import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd


class Graph:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.fps = 60
        self.totalFrames = self.fps * 3  # seconds * fps
        self.x_subset = None
        self.y_subset = None

    def setData(self, x_axis_subset, y_axis_subset):
        self.x_subset = x_axis_subset
        self.y_subset = y_axis_subset

    def animate(self, frame_num):
        print(frame_num)
        data_poins_count = len(self.x_subset["data"])
        speedMult = round(data_poins_count / self.totalFrames)

        frame_num *= speedMult

        x_data = self.x_subset["data"][: frame_num]
        y_data = self.y_subset["data"][: frame_num]

        plt.plot(x_data, y_data)

    def graph(self, animate = False, save = False):
        fig, ax = plt.subplots()

        plt.xlim([self.x_subset["min"], self.x_subset["max"]])
        plt.ylim([self.y_subset["min"], self.y_subset["max"]])

        plt.ylabel(self.y_subset["name"])
        plt.xlabel(self.x_subset["name"])

        if not animate:
            self.animate(10000) # may not be accurate
        else:
            animation = FuncAnimation(fig, func=self.animate, frames=self.totalFrames, interval=1)
            if save:
                animation.save('syncing/graph.mp4', fps=self.fps)

        if not save:
            plt.show()


csv_name = "data/SM_SbS_02_Raw_Data - SM_SoftballSwing_Trial2_Raw_Data.csv"
g = Graph(csv_name)

forcey_subset = {"data": g.df.iloc[18:10000, 2].astype(float).tolist(), "name": "ForceY", "min": -15, "max": 15}
timex_subset = {"data": g.df.iloc[18:10000, 0].astype(float).tolist(), "name": "Time", "min": 0, "max": 3}

g.setData(timex_subset, forcey_subset)
g.graph(True, True)
