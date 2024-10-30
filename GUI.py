import tkinter as tk
from tkinter import Canvas, filedialog, messagebox
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk


class VideoForceApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Video and Force Data Viewer")

        # Configure the grid to make it responsive
        self.master.columnconfigure([0, 1, 2], weight=1)  # Three columns for three screens
        self.master.rowconfigure(0, weight=1)

        # Three canvases in the first row for display
        self.canvas1 = Canvas(master, width=192, height=108, bg="black")
        self.canvas1.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.canvas2 = Canvas(master, width=200, height=200, bg="black")
        self.canvas2.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.canvas3 = Canvas(master, width=200, height=200, bg="black")
        self.canvas3.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

        # Placeholder for the image
        self.image_on_canvas1 = None
        self.image_on_canvas2 = None
        self.image_on_canvas3 = None

        # Load and display a dummy image for now
        self.current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.update_images()

        # Force label
        force_label = tk.Label(master, text="Enter force data start values:")
        force_label.grid(row=1, column=0, sticky="w", padx=5, columnspan=3)

        # Frame to contain both Entry widgets in a single row
        force_entry_frame = tk.Frame(master)
        force_entry_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5, columnspan=3)

        # Force entry for start and end in one row
        self.force_entry1 = tk.Entry(force_entry_frame, width=10)
        self.force_entry1.grid(row=0, column=0, padx=5, pady=5)

        # Video label
        video_label = tk.Label(master, text="Enter video start and end values:")
        video_label.grid(row=3, column=0, sticky="w", padx=5, columnspan=3)

        # Frame to contain both Entry widgets in a single row
        video_entry_frame = tk.Frame(master)
        video_entry_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=5, columnspan=3)

        # Video entry for start and end in one row
        self.video_entry1 = tk.Entry(video_entry_frame, width=10)
        self.video_entry1.grid(row=0, column=0, padx=5, pady=5)

        self.video_entry2 = tk.Entry(video_entry_frame, width=10)
        self.video_entry2.grid(row=0, column=1, padx=5, pady=5)

        # Add buttons to load video and force data
        load_video_button = tk.Button(master, text="Load Video", command=self.load_video)
        load_video_button.grid(row=5, column=0, sticky="ew", padx=5, pady=5, columnspan=3)

        load_force_data_button = tk.Button(master, text="Load Force Data", command=self.load_force_data)
        load_force_data_button.grid(row=6, column=0, sticky="ew", padx=5, pady=5, columnspan=3)

        # Placeholder for vector overlay display
        self.vector_overlay_label = tk.Label(master, text="Vector Overlay: None")
        self.vector_overlay_label.grid(row=7, column=0, sticky="w", padx=5, columnspan=3)

        # Initialize video capture placeholder
        self.video_capture = None
        self.fps = None
        self.frame_count = None

        # Force data attribute
        self.force_data = None
        self.rows = None

        # Booleans
        self.wrongData = None

    def check_(self):
        try:
            if self.rows is None or self.video_capture is None:
                return False
            if self.frame_count > self.rows / (600 / self.fps):
                self.wrongData = True
                return False
            return True
        except:
            pass

    def update_images(self):
        # Convert the current frame to a PhotoImage for each canvas
        img = Image.fromarray(self.current_frame)
        self.tk_image = ImageTk.PhotoImage(img)  # Create PhotoImage

        # Update each canvas with the new image
        if self.image_on_canvas1 is not None:
            self.canvas1.delete(self.image_on_canvas1)
        self.image_on_canvas1 = self.canvas1.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        if self.image_on_canvas2 is not None:
            self.canvas2.delete(self.image_on_canvas2)
        self.image_on_canvas2 = self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        if self.image_on_canvas3 is not None:
            self.canvas3.delete(self.image_on_canvas3)
        self.image_on_canvas3 = self.canvas3.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def load_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
        if video_path:
            self.video_capture = cv2.VideoCapture(video_path)
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.frame_count = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    def play_video(self):
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.update_images()
                self.master.after(30, self.play_video)  # Schedule the next frame
            else:
                self.video_capture.release()  # Release when the video ends

    def load_force_data(self):
        try:
            messagebox.showinfo("Force Data", f"Start: {self.force_entry1.get()}")
            force_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
            self.force_data = pd.read_excel(force_path, skiprows=19)
            self.rows = self.force_data.shape[0]
        except ValueError:
            messagebox.showwarning("Input Error", "Please enter valid integers for force data start and end.")


if __name__ == "__main__":
    root = tk.Tk()
    root.state('zoomed')  # Open window in full-screen mode
    app = VideoForceApp(root)
    root.mainloop()
