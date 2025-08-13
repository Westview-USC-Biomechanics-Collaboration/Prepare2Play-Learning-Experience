import tkinter as tk

class LabelManager:
    def __init__(self, parent):
        self.parent = parent  # expects a DisplayApp instance
        self.labels = {}

    def create_labels(self):
        # Label for Force Timeline
        self.labels['force_timeline'] = tk.Label(self.parent.master, text="Force Timeline (label = frame)", font=("Arial", 10))
        self.labels['force_timeline'].config(anchor="center", width=30)

        # Label for Video Timeline
        self.labels['video_timeline'] = tk.Label(self.parent.master, text="Video Timeline (label = frame)", font=("Arial", 10), bg="white")
        self.labels['video_timeline'].config(anchor="center", width=30)

        return self.labels
