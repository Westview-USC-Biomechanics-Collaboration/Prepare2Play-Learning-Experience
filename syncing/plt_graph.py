import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd


class Graph:
    def __init__(self, x_axis_subset, y_axis_subset_1, y_axis_subset_2):
        self.x_subset = x_axis_subset
        self.y_subset_1 = y_axis_subset_1
        self.y_subset_2 = y_axis_subset_2

        self.fixTimeConstraints()
        self.fixHeightConstraints()
        self.targetFPS = 60
        self.originalFPS = 2400
        print(self.x_subset['max'] - self.x_subset['min'])

        self.totalFrames = self.targetFPS * (self.x_subset['max'] - self.x_subset['min'])  # seconds * fps
        # TARGET
        # The final video will be at 60fps, and will last for the duration of self.totalFrames
        self.speedMult = round(
            len(self.x_subset['data']) / self.totalFrames
        )  # frames in original data collection / frames in target video

    def fixTimeConstraints(self):
        self.x_subset['min'] = self.x_subset['data'][0]
        self.x_subset['max'] = self.x_subset['data'][-1]

    def fixHeightConstraints(self):
        heightData = self.y_subset_2['data']

        self.y_subset_2['max'] = max(heightData) + 10
        self.y_subset_2['min'] = min(heightData) - 10

    def animation(self, frame_num):
        print(frame_num)

        frame_num *= self.speedMult

        x_data = self.x_subset["data"][: frame_num]
        y_data_1 = self.y_subset_1["data"][: frame_num]
        y_data_2 = self.y_subset_2["data"][: frame_num]

        plt.plot(x_data, y_data_1, color="g", label=self.y_subset_1["name"])
        plt.plot(x_data, y_data_2, color="r", label=self.y_subset_2["name"])
        plt.legend()

    def config_graph(self):
        plt.xlim([self.x_subset["min"], self.x_subset["max"]])
        plt.ylim([self.y_subset_2["min"], self.y_subset_2["max"]])

        plt.ylabel("Forces")
        plt.xlabel(self.x_subset["name"])

    def graph(self):
        self.config_graph()
        length = len(self.x_subset["data"])
        self.animation(length)
        plt.show()

    def animate_graph(self, save=False, saveAs=""):
        fig, ax = plt.subplots()
        self.config_graph()
        print(f"Calculating for {round(self.totalFrames)} frames")
        animation = FuncAnimation(fig, func=self.animation, frames=round(self.totalFrames), interval=1)

        if save:
            animation.save(saveAs, fps=self.targetFPS)

        else:
            plt.show()

    def getForcePlateTime(self) -> int:  # returns frame that the user steps on the forceplate
        for c, yVal in enumerate(self.y_subset_1['data']):
            if yVal >= 1:
                seconds = self.x_subset['data'][c]
                return seconds * self.targetFPS


df = pd.read_excel("D:\\USC Biomechanics Python Stuff\\Prepare2Play-Learning-Experience\\data\\Nishk\\spu_lr_NS_for01_Raw_Data_new.xlsx")

forcey_subset = {"data": df.iloc[18:10000, 1].astype(float).tolist(), "name": "ForceX", "min": -15, "max": 15}
forcez_subset = {"data": df.iloc[18:10000, 3].astype(float).tolist(), "name": "ForceZ", "min": -15, "max": 15}

timex_subset = {"data": df.iloc[18:10000, 0].astype(float).tolist(), "name": "Time", "min": 0, "max": 3}
g = Graph(timex_subset, forcey_subset, forcez_subset)
g.graph()