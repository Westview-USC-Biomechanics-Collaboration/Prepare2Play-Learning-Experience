import tkinter as tk
from tkinter import filedialog, Canvas, Label, Scale, Frame, Scrollbar, PhotoImage
import cv2
from PIL import Image, ImageTk
#from tensorflow import double
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DisplayApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Multi-Window Display App")
        self.master.geometry("1500x800")


        # Create a canvas for scrolling
        self.main_canvas = Canvas(master)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a scrollbar
        self.scrollbar = Scrollbar(master, orient="vertical", command=self.main_canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill="y")

        # Configure the canvas to work with the scrollbar
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.main_canvas.bind('<Configure>',
                              lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all")))

        # Create a frame inside the canvas to hold all widgets
        self.frame = Frame(self.main_canvas)
        self.main_canvas.create_window((0, 0), window=self.frame, anchor="nw")

        # Create three canvases for display in the first row
        self.canvas1 = Canvas(self.frame, width=400, height=300, bg="lightgrey")
        self.canvas1.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.canvas2 = Canvas(self.frame, width=400, height=300, bg="lightgrey")
        self.canvas2.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.canvas3 = Canvas(self.frame, width=400, height=300, bg="lightgrey")
        self.canvas3.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        # Create a slider in the middle row
        self.slider = Scale(self.frame, from_=0, to=100, orient="horizontal", label="Adjust Value",
                            command=self.update_slider_value)
        self.slider.grid(row=1, column=0, columnspan=3, padx=5, pady=10, sticky="ew")

        # Label to display slider value
        self.slider_value_label = Label(self.frame, text="Slider Value: 0")
        self.slider_value_label.grid(row=2, column=0, columnspan=3, pady=5)

        # Upload buttons in the bottom row
        self.upload_video_button = tk.Button(self.frame, text="Upload Video", command=self.upload_video)
        self.upload_video_button.grid(row=3, column=0, padx=5, pady=10, sticky="nsew")

        # Upload button for force data
        self.upload_force_button = tk.Button(self.frame, text="Upload force File", command=self.upload_force_data)
        self.upload_force_button.grid(row=3, column=1, padx=5, pady=10, sticky="nsew")

        # Vector overlay button
        self.show_vector_overlay = tk.Button(self.frame, text="Vector Overlay", command=lambda: print("Vector overlay clicked"))
        self.show_vector_overlay.grid(row=3, column=2, padx=5, pady=10, sticky="nsew")

        # force button
        self.video_button = tk.Button(self.frame, text="label video", command=lambda: print("Save clicked"))
        self.video_button.grid(row=4, column=0, padx=5, pady=10, sticky="nsew")

        # force button
        self.force_button = tk.Button(self.frame, text="label force", command=lambda: print("Save clicked"))
        self.force_button.grid(row=4, column=1,padx=5, pady=10, sticky="nsew")

        # Save button
        self.save_button = tk.Button(self.frame, text="Save", command=lambda: print("Save clicked"))
        self.save_button.grid(row=4, column=2,padx=5, pady=10, sticky="nsew")

        # Force timeline label
        self.force_timeline_label = Label(self.frame, text="Force Timeline (unit = frame)")
        self.force_timeline_label.grid(row=6, column=0, sticky="w")

        # Force timeline
        self.force_timeline = Canvas(self.frame, width=1080, height=75, bg="lightblue")
        self.force_timeline.grid(row=7, column=0, columnspan=3, pady=1)

        # Video timeline label
        self.video_timeline_label = Label(self.frame, text="Video Timeline (unit = frame)")
        self.video_timeline_label.grid(row=8, column=0, sticky="w")

        # Video timeline
        self.video_timeline = Canvas(self.frame, width=1080, height=75, bg="lightblue")
        self.video_timeline.grid(row=9, column=0, columnspan=3, pady=1)

        # force data
        self.force_data = None
        self.rows = None

        # video
        self.cam = None
        self.total_frames = None

        # Global fram/location base on slider
        self.loc = None


    def get_current_frame(self):
        print(self.slider.get())
        return int(self.slider.get()) # return current frame, 1st return 1

    def getSliderVal(self):
        print(self.slider.get())
        return self.slider.get()
        self.upload_csv_button = tk.Button(self.frame, text="Upload .csv", command=self.upload_csv)
        self.upload_csv_button.grid(row=3, column=1, padx=5, pady=10, sticky="ew")

        # Initialize the line reference
        self.line = None

    def update_slider_value(self, value):
        # Update the label with the current slider value
        # the line below is Ayaan's code
        #self.setVideoFrame(float(value))
        self.loc = self.get_current_frame()
        self.slider_value_label.config(text=f"Slider Value: {value}")

        # things that need to update when the slider value changes
        self.display_frame()

    def upload_video(self):
        # Open a file dialog for video files
        video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")])
        if video_path:
            print(f"Video uploaded: {video_path}")
            self.openVideo(video_path)

    def upload_force_data(self):
        # Open a file dialog for any file type
        file_path = filedialog.askopenfilename(title="Select Force Data File",filetypes=[("Excel or CSV Files", "*.xlsx;*.xls,*.csv")])
        print(f"Force data uploaded: {file_path}")
        # support both csv and excel
        if file_path.endswith('.xlsx'):
            self.forcedata = pd.read_excel(file_path,skiprows=19)
        elif file_path.endswith('.csv'):
            self.forcedata = pd.read_csv(file_path,skiprows=19)   
        self.rows = self.forcedata.shape[0]
        # Update the line position based on slider value if the line exists
        if self.line:
            max_val = self.slider['to']  # Maximum slider value
            normalized_position = int(value) / max_val
            x_position = self.ax.get_xlim()[0] + normalized_position * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
            self.line.set_xdata([x_position, x_position])
            self.canvas.draw()

    def upload_csv(self):
        # Open a file dialog for CSV files
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                # Load the CSV file starting from row 20, with the first two columns as x and y
                data = pd.read_csv(file_path, skiprows=19, header=None)
                x = data.iloc[:, 0]
                y = data.iloc[:, 1]
                
                # Plot the data
                self.plot_csv_data(x, y)
            except Exception as e:
                print(f"Error loading CSV: {e}")

    def openVideo(self, video_path):
        print("set1")
        self.cam = cv2.VideoCapture(video_path)
        self.total_frames = self.cam.get(cv2.CAP_PROP_FRAME_COUNT)

        # print(total_frames)
        self.slider.config(to=self.total_frames)
        #self.cam.set(cv2.CAP_PROP_POS_FRAMES, 600) # set the total amount of frame to 600 for now   

    def display_frame(self):
        self.cam.set(cv2.CAP_PROP_POS_FRAMES, self.loc-1) # pick current frame index 1st is 0
        ret, frame = self.cam.read()

        if ret:
            frame = Image.fromarray(frame).resize((400, 300), resample=Image.BICUBIC) # Resize the frame to 400 * 300
            self.photo_image1 = ImageTk.PhotoImage(frame)
            self.canvas1.create_image(0, 0, image=self.photo_image1, anchor=tk.NW)
    """
    # Ayaan's code for displaying the frame
    def setVideoFrame(self, frameNum: double):
        self.canvas1.delete("all")
        self.cam.set(cv2.CAP_PROP_FRAME_COUNT, frameNum)
        ret, frame = self.cam.read()

        if not ret: return
        frame = Image.fromarray(frame).resize((400, 300), resample=Image.BICUBIC)
        self.photo_image1 = ImageTk.PhotoImage(frame)

        self.canvas1.create_image(0, 0, image=self.photo_image1, anchor=tk.NW)
    """
    def display_image(self, file_path):
        # Load and resize the image using Pillow
        image = Image.open(file_path)
        image = image.resize((400, 300), resample=Image.BICUBIC)
    def plot_csv_data(self, x, y):
        # Clear previous figure on canvas2
        for widget in self.canvas2.winfo_children():
            widget.destroy()

        # Create a new figure and plot
        self.fig, self.ax = plt.subplots(figsize=(4.75, 3.75))
        self.ax.plot(x, y, marker='o', linestyle='-', color='blue', linewidth = 0.5)
        self.ax.set_title("Force vs. Time")
        self.ax.set_xlabel("Force (N.)")
        self.ax.set_ylabel("Time (s.)")

        # Draw an initial vertical line on the left
        self.line = self.ax.axvline(x=x.iloc[0], color='red', linestyle='--', linewidth=1.5)

        # Embed the matplotlib figure in the Tkinter canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.canvas2)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()



if __name__ == "__main__":
    root = tk.Tk()
    app = DisplayApp(root)
    root.mainloop()
