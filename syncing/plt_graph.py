import matplotlib.pyplot as plt
import matplotlib.animation as anim
import pandas as pd


class Graph:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

    def graph(self, y_axis: {}):
        timex_subset = self.df.iloc[18:10000, 0].astype(float).tolist()

        plt.plot(timex_subset, y_axis["data"])
        plt.ylabel(y_axis["name"])
        plt.xlabel("Time")
        plt.xlim([0, 3])
        plt.ylim([y_axis["min"], y_axis["max"]])

        plt.show()


g = Graph("data/SM_SbS_02_Raw_Data - SM_SoftballSwing_Trial2_Raw_Data.csv")
# Select subset of data for plotting
forcey_subset = {"data": g.df.iloc[18:10000, 2].astype(float).tolist(), "name": "ForceY", "min": -35, "max": 35}
g.graph(forcey_subset)