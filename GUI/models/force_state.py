import pandas as pd

class ForceState:
    def __init__(self):
        self.path = None
        self.data = pd.DataFrame()
        self.rows = 0

    def load(self, path):
        self.path = path
        self.data = pd.read_csv(path, header=17, delimiter="\t").drop(0)
        self.rows = len(self.data)
