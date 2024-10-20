import tkinter as tk
from tkinter import Canvas, PhotoImage, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk


class VideoForceApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Video and Force Data Viewer")

        # Create a canvas to display the video
        self.canvas = Canvas(master, width=640, height=480)
        self.canvas.pack()

        # Placeholder for the image
        self.image_on_canvas = None

        # Load and display a dummy image for now
        self.current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.update_image()

        # Create timelines
        self.video_timeline = Canvas(master, height=50, bg="lightblue")
        self.video_timeline.pack(fill=tk.X)

        self.force_data_timeline = Canvas(master, height=50, bg="lightgreen")
        self.force_data_timeline.pack(fill=tk.X)

        # Add buttons to load video and force data
        load_video_button = tk.Button(master, text="Load Video", command=self.load_video)
        load_video_button.pack()

        load_force_data_button = tk.Button(master, text="Load Force Data", command=self.load_force_data)
        load_force_data_button.pack()

        # Placeholder for vector overlay display
        self.vector_overlay_label = tk.Label(master, text="Vector Overlay: None")
        self.vector_overlay_label.pack()

    def update_image(self):
        # Convert the current frame to a PhotoImage
        img = Image.fromarray(self.current_frame)
        self.tk_image = ImageTk.PhotoImage(img)  # Create PhotoImage

        # Update the canvas with the new image
        if self.image_on_canvas is not None:
            self.canvas.delete(self.image_on_canvas)  # Remove previous image
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def load_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
        if video_path:
            self.video_capture = cv2.VideoCapture(video_path)
            self.play_video()

    def play_video(self):
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.update_image()
            self.master.after(30, self.play_video)  # Schedule the next frame

    def load_force_data(self):
        force_data_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if force_data_path:
            # Placeholder for loading and processing force data
            print(f"Loaded force data from {force_data_path}")
            self.vector_overlay_label.config(text="Vector Overlay: Loaded Force Data")


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoForceApp(root)
    root.mainloop()
