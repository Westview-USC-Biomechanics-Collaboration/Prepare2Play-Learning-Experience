import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Graph:
    def __init__(self, x_axis_subset, y_axis_subset):
        self.x_subset = x_axis_subset
        self.y_subset = y_axis_subset
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
        heightData = self.y_subset['data']

        self.y_subset['max'] = max(heightData) + 10
        self.y_subset['min'] = min(heightData) - 10

    def animation(self, frame_num):
        print(frame_num)

        frame_num *= self.speedMult

        x_data = self.x_subset["data"][: frame_num]
        y_data = self.y_subset["data"][: frame_num]

        plt.plot(x_data, y_data)

    def config_graph(self):
        plt.xlim([self.x_subset["min"], self.x_subset["max"]])
        plt.ylim([self.y_subset["min"], self.y_subset["max"]])

        plt.ylabel(self.y_subset["name"])
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
        for c, yVal in enumerate(self.y_subset['data']):
            if yVal >= 1:
                seconds = self.x_subset['data'][c]
                return seconds * self.targetFPS

