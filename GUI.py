import tkinter as tk
from tkinter import filedialog, Canvas, Label, Scale, Frame, Scrollbar, PhotoImage
import cv2
from PIL import Image, ImageTk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# our script
from Timeline import timeline
from vector_overlay import vectoroverlay_GUI

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


        """
        Deren's code (35~37) for putting frame and canvas together. I don't know how it works but it works
        """
        # Create a frame inside the canvas to hold all widgets
        self.frame = Frame(self.main_canvas)
        self.main_canvas.create_window((0, 0), window=self.frame, anchor="nw")

        # Create three canvases for display in the first row
        self.canvas1 = Canvas(self.frame, width=400, height=300, bg="lightgrey")
        self.canvas1.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.photo_image1 = None # place holder for the image object

        self.canvas2 = Canvas(self.frame, width=400, height=300, bg="lightgrey")
        self.canvas2.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.photo_image2 = None 

        self.canvas3 = Canvas(self.frame, width=400, height=300, bg="lightgrey")
        self.canvas3.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        self.photo_image3 = None 

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
        self.show_vector_overlay = tk.Button(self.frame, text="Vector Overlay", command=self.vector_overlay)
        self.show_vector_overlay.grid(row=3, column=2, padx=5, pady=10, sticky="nsew")

        # video label button
        self.video_button = tk.Button(self.frame, text="label video", command=self.label_video)
        self.video_button.grid(row=4, column=0, padx=5, pady=10, sticky="nsew")

        # force label button
        self.force_button = tk.Button(self.frame, text="label force", command=self.label_force)
        self.force_button.grid(row=4, column=1,padx=5, pady=10, sticky="nsew")

        # Save button
        self.save_button = tk.Button(self.frame, text="Save", command=self.save)
        self.save_button.grid(row=4, column=2,padx=5, pady=10, sticky="nsew")

        # Force timeline label
        self.force_timeline_label = Label(self.frame, text="Force Timeline (unit = frame)")
        self.force_timeline_label.grid(row=6, column=0, sticky="w")

        # Force timeline
        self.force_timeline = Canvas(self.frame, width=1080, height=75, bg="lightblue")
        self.force_timeline.grid(row=7, column=0, columnspan=3, pady=1)
        self.timeline_image1 = None  # place holder for timeline cavas image object

        # Video timeline label
        self.video_timeline_label = Label(self.frame, text="Video Timeline (unit = frame)")
        self.video_timeline_label.grid(row=8, column=0, sticky="w")

        # Video timeline
        self.video_timeline = Canvas(self.frame, width=1080, height=75, bg="lightblue")
        self.video_timeline.grid(row=9, column=0, columnspan=3, pady=1)
        self.timeline_image2 = None

        # align botton 
        self.align_button = tk.Button(self.frame, text="Align", command=self.align)
        self.align_button.grid(row=10,column=0,padx=5,pady=10,sticky="nsew")

        # force data
        self.force_path = None
        self.force_data = None
        self.rows = None
        self.force_frame = None   # convert rows to frames

        # Graph
        self.x = None # x-axis data
        self.y = None # y-axis data
        self.line = None # Initialize the line reference
        self.canvas = None # the widget for matplot

        # video
        self.video_path = None
        self.cam = None
        self.total_frames = None

        # timeline
        self.timeline1 = None
        self.timeline2 = None

        # labels
        self.force_align = None
        self.video_align = None

        # Global frame/location base on slider
        self.loc = 0



    def get_current_frame(self):
        print(self.slider.get())
        return int(self.slider.get()) # return current frame, 1st return 1

    """
    ################## 
    Below is the method that is run everytime the user update the slider value
    be sure to put everything you want to run under this method
    ################## 
    """
    def update_slider_value(self, value):

        # Update the label with the current slider value
        self.loc = self.get_current_frame()
        self.slider_value_label.config(text=f"Slider Value: {value}")

        # Things that need to be updated when the slider value changes

        if self.cam:
            # draw video canvas
            self.display_frame()

            # update video timeline
            videoTimeline = Image.fromarray(self.timeline2.draw_rect(loc=self.loc / self.total_frames))
            self.timeline_image2 = ImageTk.PhotoImage(videoTimeline)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            self.video_timeline.create_image(0, 0, image=self.timeline_image2, anchor=tk.NW)

        if self.rows is not None:  # somehow self.forcedata is not None doesn't work, using self.rows as compensation
            # draw graph canvas
            normalized_position = int(value) / (self.slider['to'])
            x_position = self.ax.get_xlim()[0] + normalized_position * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
            self.line.set_xdata([x_position, x_position])
            self.canvas.draw()

            # update force timeline
            forceTimeline = Image.fromarray(self.timeline1.draw_rect(loc=self.loc / self.slider['to']))
            self.timeline_image1 = ImageTk.PhotoImage(forceTimeline)
            self.force_timeline.create_image(0, 0, image=self.timeline_image1, anchor=tk.NW)




    def upload_video(self):
        # Open a file dialog for video files
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")])
        if self.video_path:
            print(f"Video uploaded: {self.video_path}")
            # display video
            self.openVideo(self.video_path)

            # Initialize video timeline
            self.timeline2 = timeline(0,1)
            videoTimeline = Image.fromarray(self.timeline2.draw_rect(loc=self.loc))
            self.timeline_image2 = ImageTk.PhotoImage(videoTimeline)   # create image object that canvas object accept
            self.video_timeline.create_image(0, 0, image=self.timeline_image2, anchor=tk.NW)

    def upload_force_data(self):
        # Open a file dialog for any file type
        file_path = filedialog.askopenfilename(title="Select Force Data File",filetypes=[("Excel or CSV Files", "*.xlsx;*.xls,*.csv")])
        self.force_path = file_path
        print(f"Force data uploaded: {file_path}")
        # support both csv and excel
        if file_path.endswith('.xlsx'):
            self.forcedata = pd.read_excel(file_path,skiprows=19)   # ---> skip useless rows
        elif file_path.endswith('.csv'):
            self.forcedata = pd.read_csv(file_path,skiprows=19)   
        self.rows = self.forcedata.shape[0]
        self.force_frame = int(self.rows/(600/self.cam.get(cv2.CAP_PROP_FPS)))  # assume fix step size (10)---> result is num of frames(int)

        self.x = self.forcedata.iloc[:, 0] # time
        self.y = self.forcedata.iloc[:, 1] # force x   ---> we need 2 radio button for picking the force place and 3 radio button to pick the force
        self.plot_force_data()

        # Initialize force timeline
        print(f"force frame: {self.force_frame}")
        # create a timeline object, defining end as (num of frame in forcedata /  max slider value)
        # Slider value should be updated to frame count when user upload the video file,
        # otherwise we will use the default slider value(100).
        self.timeline1 = timeline(0,self.force_frame/self.slider['to'])
        forceTimeline = Image.fromarray(self.timeline1.draw_rect(loc=self.loc))
        self.timeline_image1 = ImageTk.PhotoImage(forceTimeline)  # create image object that canvas object accept
        self.force_timeline.create_image(0, 0, image=self.timeline_image1, anchor=tk.NW)



    def openVideo(self, video_path):
        self.cam = cv2.VideoCapture(video_path)
        self.total_frames = self.cam.get(cv2.CAP_PROP_FRAME_COUNT)
        self.slider.config(to=self.total_frames)   # ---> reconfigure slider value. The max value is the total number of frame in the video
        self.display_frame()

    def display_frame(self):
        self.cam.set(cv2.CAP_PROP_POS_FRAMES, self.loc-1) # pick the corresponding frame to display || the 1st frame is index 0, therefore -1
        ret, frame = self.cam.read()  # the `frame` object is now the frame we want

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            frame = Image.fromarray(frame).resize((400, 300), resample=Image.BICUBIC) # Resize the frame to 400 * 300
            self.photo_image1 = ImageTk.PhotoImage(frame)   # ---> update the image object base on current frame.
            self.canvas1.create_image(0, 0, image=self.photo_image1, anchor=tk.NW)

    def plot_force_data(self):
        # Clear previous figure on canvas2
        for widget in self.canvas2.winfo_children():
            widget.destroy()

        # Create a new figure and plot
        self.fig, self.ax = plt.subplots(figsize=(4.75, 3.75))
        self.ax.plot(self.x, self.y, linestyle='-', color='blue', linewidth = 0.5)
        self.ax.set_title("Force vs. Time")
        self.ax.set_xlabel("Force (N.)")
        self.ax.set_ylabel("Time (s.)")

        # Draw an initial vertical line on the left
        self.line = self.ax.axvline(x=self.x.iloc[0], color='red', linestyle='--', linewidth=1.5)

        # Embed the matplotlib figure in the Tkinter canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.canvas2)   # ---> self.canvas holds the object that represent image on canvas, I'm not too sure about this.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

    def label_force(self):  # ---> executed when user click label force
        self.force_align = self.loc
        self.timeline1.update_label(self.loc/self.slider['to'])
    def label_video(self):  # ---> executed when user click label video
        self.video_align = self.loc
        self.timeline2.update_label(self.loc/self.slider['to'])

    """
    The alignment method has a problem. The user can only use it once.
    If the user use the align button twice, self.force_align lost true frame value relative to global
    meaning that we are not able to convert self.force_align to the correct row in force data
    This can be solve by adding a new variable that contain the labeled row.
    """
    def align(self):
        print("User clicked align button")
        print(self.force_align, self.video_align)

        # update the timeline visually
        start, end = self.timeline1.get_start_end()
        offset = self.force_align - self.video_align
        newstart = start-offset/self.slider['to']
        newend = end-offset/self.slider['to']
        newlabel = self.timeline1.get_label()-offset/self.slider['to']
        print(f"new start percentage: {newstart}\nnew end percentage: {newend}")
        self.timeline1.update_start_end(newstart,newend)
        self.timeline1.update_label(newlabel)

    def save(self):
        """
        Assuming there is a labeled row value.
        """
        print("user clicked save button")
        pass

    def vector_overlay(self):
        print("user clicked vector overlay button")

        v = vectoroverlay_GUI.VectorOverlay(data=self.forcedata,video=self.cam)
        v.LongVectorOverlay(outputName="C:\\Users\\16199\Desktop\data\Chase\\testoutput.mp4")





if __name__ == "__main__":
    root = tk.Tk()
    app = DisplayApp(root)
    root.mainloop()
