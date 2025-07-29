import tkinter as tk
from PIL import Image, ImageTk
from GUI.Timeline import timeline

class TimelineManager:
    def __init__(self, parent):
        self.parent = parent
        self.timeline1 = None
        self.timeline2 = None
        self.force_canvas = None
        self.video_canvas = None
        self.timeline_image1 = None
        self.timeline_image2 = None

    def create_timelines(self):
        self.force_canvas = tk.Canvas(self.parent.master, width=1080, height=75, bg="lightblue")
        self.video_canvas = tk.Canvas(self.parent.master, width=1080, height=75, bg="lightblue")

        return self.force_canvas, self.video_canvas

    def initialize_force_timeline(self, force_frame):
        self.timeline1 = timeline(0, force_frame / self.parent.slider['to'])
        img = Image.fromarray(self.timeline1.draw_rect(loc=self.parent.state.loc))

        resized = img.resize(
            (self.force_canvas.winfo_width(), self.force_canvas.winfo_height()),
            Image.Resampling.LANCZOS
        )
        self.timeline_image1 = ImageTk.PhotoImage(resized)
        self.force_canvas.create_image(0, 0, image=self.timeline_image1, anchor=tk.NW)

    def initialize_video_timeline(self):
        self.timeline2 = timeline(0, 1)  # Start with dummy timeline
        self.update_video_timeline()

    def update_video_timeline(self):
        img = Image.fromarray(self.timeline2.draw_rect(loc=self.parent.state.loc / self.parent.Video.total_frames))

        resized = img.resize(
            (self.video_canvas.winfo_width(), self.video_canvas.winfo_height()),
            Image.Resampling.LANCZOS
        )
        self.timeline_image2 = ImageTk.PhotoImage(resized)
        self.video_canvas.create_image(0, 0, image=self.timeline_image2, anchor=tk.NW)

    def update_force_timeline(self):
        img = Image.fromarray(self.timeline1.draw_rect(loc=self.parent.state.loc / self.parent.slider['to']))

        resized = img.resize(
            (self.force_canvas.winfo_width(), self.force_canvas.winfo_height()),
            Image.Resampling.LANCZOS
        )
        self.timeline_image1 = ImageTk.PhotoImage(resized)
        self.force_canvas.create_image(0, 0, image=self.timeline_image1, anchor=tk.NW)
