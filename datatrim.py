import tkinter as tk
from tkinter import Canvas, PhotoImage, filedialog, messagebox
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

class VideoForceApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Video and Force Data Viewer")

        # Create a canvas to display the video
        self.canvas = Canvas(master, width=640, height=360)
        self.canvas.pack()

        # Placeholder for the image
        self.image_on_canvas = None

        # Load and display a dummy image for now
        self.current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.update_image()

        # Force label
        force_label = tk.Label(master, text="Enter force data start values:")
        force_label.pack()

        # Frame to contain both Entry widgets in a single row
        force_entry_frame = tk.Frame(master)
        force_entry_frame.pack()

        # Force entry for start and end in one row
        self.force_entry1 = tk.Entry(force_entry_frame, width=10)
        self.force_entry1.grid(row=0, column=0, padx=5, pady=5)


        # video label
        force_label = tk.Label(master, text="Enter video start and end values:")
        force_label.pack()

        # Frame to contain both Entry widgets in a single row
        video_entry_frame = tk.Frame(master)
        video_entry_frame.pack()

        # Force entry for start and end in one row
        self.video_entry1 = tk.Entry(video_entry_frame, width=10)
        self.video_entry1.grid(row=0, column=0, padx=5, pady=5)

        self.video_entry2 = tk.Entry(video_entry_frame, width=10)
        self.video_entry2.grid(row=0, column=1, padx=5, pady=5)

        # Add buttons to load video and force data
        load_video_button = tk.Button(master, text="Load Video", command=self.load_video)
        load_video_button.pack()

        load_force_data_button = tk.Button(master, text="Load Force Data", command=self.load_force_data)
        load_force_data_button.pack()

        # Placeholder for vector overlay display
        self.vector_overlay_label = tk.Label(master, text="Vector Overlay: None")
        self.vector_overlay_label.pack()

        # Initialize video capture placeholder
        self.video_capture = None
        self.fps = None
        self.frame_count = None

        # force data attribute
        self.force_data = None
        self.rows = None

        # booleans
        self.wrongData = None

    def check_(self):
        try:
            if(self.rows==None  or  self.video_capture==None):
                return False
            if(self.frame_count>self.rows/(600/self.fps)):
                self.wrongData = True
                return False
            return True
        except:
            pass
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
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.frame_count = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            #self.play_video()

    def play_video(self):
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.update_image()
                self.master.after(30, self.play_video)  # Schedule the next frame
            else:
                self.video_capture.release()  # Release when the video ends

    def scale_down(self):
        pass

    def load_force_data(self):
        try:
            messagebox.showinfo("Force Data", f"Start: {self.force_entry1}")
            # You can process the force data here or load a file
            force_path = filedialog.askopenfilename(filetypes=[("force files", "*xslx")])
            self.force_data = pd.read_excel(force_path,skiprows=19)
            self.rows = self.force_data.shape[0]

            # Placeholder for further processing of force data
            #self.vector_overlay_label.config(text="Vector Overlay: Force Data Loaded")
        except ValueError:
            messagebox.showwarning("Input Error", "Please enter valid integers for force data start and end.")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoForceApp(root)
    root.mainloop()
