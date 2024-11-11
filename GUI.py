import tkinter as tk
from tkinter import filedialog, Canvas, Label, Scale, Frame, Scrollbar, PhotoImage
import cv2
from PIL import Image, ImageTk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import shutil
# our script
from Timeline import timeline
from vector_overlay import vectoroverlay_GUI


class DisplayApp:
    def __init__(self, master):
        self.master = master
        # Give direction
        direction = (
            "This is the prototype syncing app. Please following the directions given, otherwise it won't work.\n"
            "REQUIREMENT:\n"
            "a)Video start at the moment when tennis ball collides force plate"
            "\n\n"
            "Step 1: upload video\n"
            "Step 2: upload forcedata\n"
            "Step 3: click `label video` when slider value is 0\n"
            "Step 4: drag the slider to find the force spike on the graph\n"
            "Step 5: click `label force`\n"
            "Step 6: click align, you may need to extend the window to see that button\n"
            "Step 7: click `vector overlay` button\n"
            "Step 8: click `save` button and set the output name")
        self.pop_up(text=direction)

        self.master.title("Multi-Window Display App")
        self.master.geometry("1500x800")
        self.master.lift()


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

        """
        Row 0
        """
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

        """
        Row 1
        """
        # align botton
        self.align_button = tk.Button(self.frame, text="Align", command=self.align)
        self.align_button.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.graph_option = tk.Button(self.frame,text="graphing options",command=self.graph)
        self.graph_option.grid(row=1,column = 1,padx=10,pady=10,sticky="nsew")

        """
        Row 2
        """
        # Create a slider in the middle row
        self.slider = Scale(self.frame, from_=0, to=100, orient="horizontal", label="Adjust Value",
                            command=self.update_slider_value)
        self.slider.grid(row=2, column=0, columnspan=3, padx=5, pady=10, sticky="ew")

        """
        Row 3
        """
        # Label to display slider value
        self.slider_value_label = Label(self.frame, text="Slider Value: 0")
        self.slider_value_label.grid(row=3, column=0, columnspan=3, pady=5)

        """
        Row 4
        """

        # Upload buttons in the bottom row
        self.upload_video_button = tk.Button(self.frame, text="Upload Video", command=self.upload_video)
        self.upload_video_button.grid(row=4, column=0, padx=5, pady=10, sticky="nsew")

        # Upload button for force data
        self.upload_force_button = tk.Button(self.frame, text="Upload force File", command=self.upload_force_data)
        self.upload_force_button.grid(row=4, column=1, padx=5, pady=10, sticky="nsew")

        # Vector overlay button
        self.show_vector_overlay = tk.Button(self.frame, text="Vector Overlay", command=self.vector_overlay)
        self.show_vector_overlay.grid(row=4, column=2, padx=5, pady=10, sticky="nsew")

        """
        Row 5
        """
        # video label button
        self.video_button = tk.Button(self.frame, text="label video", command=self.label_video)
        self.video_button.grid(row=5, column=0, padx=5, pady=10, sticky="nsew")

        # force label button
        self.force_button = tk.Button(self.frame, text="label force", command=self.label_force)
        self.force_button.grid(row=5, column=1,padx=5, pady=10, sticky="nsew")

        # Save button
        self.save_button = tk.Button(self.frame, text="Save", command=self.save)
        self.save_button.grid(row=5, column=2,padx=5, pady=10, sticky="nsew")

        """
        Row 6
        """
        # Force timeline label
        self.force_timeline_label = Label(self.frame, text="Force Timeline (unit = frame)")
        self.force_timeline_label.grid(row=7, column=0, sticky="w")

        """
        Row 7
        """
        # Force timeline
        self.force_timeline = Canvas(self.frame, width=1080, height=75, bg="lightblue")
        self.force_timeline.grid(row=8, column=0, columnspan=3, pady=1)
        self.timeline_image1 = None  # place holder for timeline cavas image object

        """
        Row 8
        """
        # Video timeline label
        self.video_timeline_label = Label(self.frame, text="Video Timeline (unit = frame)")
        self.video_timeline_label.grid(row=9, column=0, sticky="w")

        """
        Row 9
        """
        # Video timeline
        self.video_timeline = Canvas(self.frame, width=1080, height=75, bg="lightblue")
        self.video_timeline.grid(row=10, column=0, columnspan=3, pady=1)
        self.timeline_image2 = None

        # force data
        self.force_path = None
        self.force_data = None
        self.rows = None
        self.force_frame = None   # convert rows to frames
        self.step_size = None

        # Graph
        self.x = None # x-axis data
        self.y = None # y-axis data
        self.line = None # Initialize the line reference
        self.canvas = None # the widget for matplot

        # Graphing options
        self.plate = tk.StringVar(value="Force Plate 1")
        self.force = tk.StringVar(value="Fz")

        # video
        self.video_path = None
        self.cam = None
        self.total_frames = None

        self.vector_cam = None

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

    def pop_up(self, text):
        # Create a new top-level window (popup)
        popup = tk.Toplevel(self.master)
        popup.title("Popup Window")

        # Set the geometry of the popup window (500x300 in this case)
        popup.geometry("500x300")

        # Optionally, center the popup relative to the parent window
        # Get the screen width and height
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()

        # Get the dimensions of the popup
        popup_width = 500
        popup_height = 300

        # Calculate the position to center the popup
        position_top = int(screen_height / 2 - popup_height / 2)
        position_right = int(screen_width / 2 - popup_width / 2)

        # Set the geometry with calculated position
        popup.geometry(f'{popup_width}x{popup_height}+{position_right}+{position_top}')

        # Add label to the popup window
        label = tk.Label(popup, text=text)
        label.pack(pady=20)

        # Add a close button to the popup window
        close_button = tk.Button(popup, text="Close", command=popup.destroy)
        close_button.pack()

        # Ensure the popup stays above the main window
        popup.lift()  # Bring the popup to the front

        # Make the popup modal (blocks interaction with the main window)
        popup.grab_set()

        # Wait for the popup to be destroyed before returning to the main window
        self.master.wait_window(popup)

    def update_force_timeline(self):
        forceTimeline = Image.fromarray(self.timeline1.draw_rect(loc=self.loc / self.slider['to']))
        self.timeline_image1 = ImageTk.PhotoImage(forceTimeline)
        self.force_timeline.create_image(0, 0, image=self.timeline_image1, anchor=tk.NW)

    def update_video_timeline(self):
        videoTimeline = Image.fromarray(self.timeline2.draw_rect(loc=self.loc / self.total_frames))
        self.timeline_image2 = ImageTk.PhotoImage(videoTimeline)
        self.video_timeline.create_image(0, 0, image=self.timeline_image2, anchor=tk.NW)

    def openVideo(self, video_path):
        self.cam = cv2.VideoCapture(video_path)
        self.total_frames = self.cam.get(cv2.CAP_PROP_FRAME_COUNT)
        self.slider.config(to=self.total_frames)   # ---> reconfigure slider value. The max value is the total number of frame in the video
        self.photo_image1 = self.display_frame(camera=self.cam)
        self.canvas1.create_image(0, 0, image=self.photo_image1, anchor=tk.NW)

    def display_frame(self,camera):
        camera.set(cv2.CAP_PROP_POS_FRAMES, self.loc) # pick the corresponding frame to display || the 1st frame is index 0, therefore -1
        ret, frame = camera.read()  # the `frame` object is now the frame we want

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            frame = Image.fromarray(frame).resize((400, 300), resample=Image.BICUBIC) # Resize the frame to 400 * 300
            photoImage = ImageTk.PhotoImage(frame)   # ---> update the image object base on current frame.
            return photoImage

    def plot_force_data(self):
        # Clear previous figure on canvas2
        for widget in self.canvas2.winfo_children():
            widget.destroy()

        # Create a new figure and plot
        self.fig, self.ax = plt.subplots(figsize=(4.75, 3.75))

        print(self.plate,self.force)
        if self.plate.get()=="Force Plate 1":
            if self.force.get()=="Fx":
                self.y=self.force_data.loc[:,"Fx1"]
            elif self.force.get()=="Fy":
                self.y=self.force_data.loc[:,"Fy1"]
            elif self.force.get()=="Fz":
                self.y=self.force_data.loc[:,"Fz1"]
        elif self.plate.get()=="Force Plate 2":
            if self.force.get()=="Fx":
                self.y=self.force_data.loc[:,"Fx2"]
            elif self.force.get()=="Fy":
                self.y=self.force_data.loc[:,"Fy2"]
            elif self.force.get()=="Fz":
                self.y=self.force_data.loc[:,"Fz2"]

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

    """
    # methods above are functions
    
    The alignment method has a problem. The user can only use it once.
    If the user use the align button twice, self.force_align lost true frame value relative to global
    meaning that we are not able to convert self.force_align to the correct row in force data
    This can be solve by adding a new variable that contain the labeled row.
    
    # methods below are buttons and slider that user can interact with
    """
    def update_slider_value(self, value):

        # Update the label with the current slider value
        self.loc = self.get_current_frame()
        self.slider_value_label.config(text=f"Slider Value: {value}")

        # Things that need to be updated when the slider value changes

        if self.cam:
            # draw video canvas
            self.photo_image1 = self.display_frame(camera=self.cam)
            self.canvas1.create_image(0, 0, image=self.photo_image1, anchor=tk.NW)
            # update video timeline
            self.update_video_timeline()
        if self.vector_cam:
            # draw vector overlay canvas
            self.photo_image3 = self.display_frame(camera=self.vector_cam)
            self.canvas3.create_image(0, 0, image=self.photo_image3, anchor=tk.NW)




        if self.rows is not None:  # somehow self.force_data is not None doesn't work, using self.rows as compensation
            # draw graph canvas
            # normalized_position = int(value) / (self.slider['to'])
            # x_position = self.ax.get_xlim()[0] + normalized_position * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
            try:
                x_position = float(self.force_data.iloc[int(self.loc * self.step_size),0])
                self.line.set_xdata([x_position])
                self.canvas.draw()
            except IndexError as e:
                print("force data out of range")

            # update force timeline
            self.update_force_timeline()

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

        names = ["abs time (s)", "Fx1", "Fy1", "Fz1", "|Ft1|", "Ax1", "Ay1", "COM px1", "COM py1", "COM pz1",
                 "Fx2", "Fy2", "Fz2", "|Ft2|", "Ax2", "Ay2", "COM px2", "COM py2", "COM pz2"]
        # support both csv and excel
        if file_path.endswith('.xlsx'):
            self.force_data = pd.read_excel(
                file_path,
            )
        elif file_path.endswith('.csv'):
            self.force_data = pd.read_csv(
                file_path
            )

        self.force_data = self.force_data.iloc[18:,0:len(names)].reset_index(drop=True)
        self.force_data.columns = names
        self.force_data = self.force_data.apply(pd.to_numeric, errors='coerce')



        self.rows = self.force_data.shape[0]
        try:
            self.step_size = (600/self.cam.get(cv2.CAP_PROP_FPS)) # rows/frame
        except AttributeError as e:
            print("Video file missing!!!\nProceeding assuming step size is 20 rows/frame")
            self.pop_up("Video file missing!!!\n\nProceeding assuming step size is 20 rows/frame\n\n"
                        "Please reload the force data after uploading the video")
            self.step_size = 20
        self.force_frame = int(self.rows/self.step_size)  # represent num of frames force data can cover

        self.x = self.force_data.iloc[:, 0] # time
        self.y = self.force_data.iloc[:, 3] # force z   ---> we need 2 radio button for picking the force place and 3 radio button to pick the force
        self.plot_force_data()

        # Initialize force timeline
        print(f"force frame: {self.force_frame}")
        """
        # create a timeline object, defining end as (num of frame in force_data /  max slider value)
        # Slider value should be updated to frame count when user upload the video file,
        # otherwise we will use the default slider value(100).
        """
        self.timeline1 = timeline(0,self.force_frame/self.slider['to'])
        forceTimeline = Image.fromarray(self.timeline1.draw_rect(loc=self.loc))
        self.timeline_image1 = ImageTk.PhotoImage(forceTimeline)  # create image object that canvas object accept
        self.force_timeline.create_image(0, 0, image=self.timeline_image1, anchor=tk.NW)

    def align(self):
        print("User clicked align button")
        print(self.force_align, self.video_align)

        # update the timeline visually
        start, end = self.timeline1.get_start_end()
        try:
            offset = self.force_align - self.video_align
            newstart = start-offset/self.slider['to']
            newend = end-offset/self.slider['to']
            newlabel = self.timeline1.get_label()-offset/self.slider['to']
            print(f"new start percentage: {newstart}\nnew end percentage: {newend}")
            self.timeline1.update_start_end(newstart,newend)
            self.timeline1.update_label(newlabel)

            self.force_data = self.force_data[int(self.force_align*self.step_size):]
            print("cut force data")
            self.slider.set(0)
            self.plot_force_data()
        except TypeError as e:
            self.pop_up("Missing label!!!")
            print("missing label")

    def save(self):
        print("user clicked save button")
        file_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",  # Default extension if none is provided
            filetypes=[("MP4 file", "*.mp4"), ("All files", "*.*")]
        )
        shutil.copy("vector_overlay_temp.mp4",file_path)
        os.remove("vector_overlay_temp.mp4")

    def vector_overlay(self):
        print("user clicked vector overlay button")
        temp_video = "vector_overlay_temp.mp4"
        v = vectoroverlay_GUI.VectorOverlay(data=self.force_data,video=self.cam)
        v.LongVectorOverlay(outputName=temp_video)

        self.vector_cam = cv2.VideoCapture(temp_video)

    def label_force(self):  # ---> executed when user click label force
        self.force_align = self.loc
        self.timeline1.update_label(self.loc/self.slider['to'])
        self.update_force_timeline()

    def label_video(self):  # ---> executed when user click label video
        self.video_align = self.loc
        self.timeline2.update_label(self.loc/self.slider['to'])
        self.update_video_timeline()

    def graph(self):
        # Create a new popup window
        popup = tk.Toplevel(self.frame)
        popup.title("Force Plate Selection")
        popup.geometry("300x250")

        # Variables to store selected radio button values
        self.plate = tk.StringVar(value="Force Plate 1")
        self.force = tk.StringVar(value="Fx")

        # First row: Force Plate Selection
        frame1 = tk.Frame(popup)
        frame1.pack(pady=10)

        tk.Label(frame1, text="Select Force Plate:").pack(side=tk.LEFT)
        force_plate_1 = tk.Radiobutton(frame1, text="Force Plate 1", variable=self.plate,
                                       value="Force Plate 1")
        force_plate_1.pack(side=tk.LEFT)

        force_plate_2 = tk.Radiobutton(frame1, text="Force Plate 2", variable=self.plate,
                                       value="Force Plate 2")
        force_plate_2.pack(side=tk.LEFT)

        # Second row: Force Components
        frame2 = tk.Frame(popup)
        frame2.pack(pady=10)

        tk.Label(frame2, text="Select Force").pack()

        fx_radio = tk.Radiobutton(frame2, text="Fx", variable=self.force, value="Fx")
        fx_radio.pack(side=tk.LEFT, padx=5)

        fy_radio = tk.Radiobutton(frame2, text="Fy", variable=self.force, value="Fy")
        fy_radio.pack(side=tk.LEFT, padx=5)

        fz_radio = tk.Radiobutton(frame2, text="Fz", variable=self.force, value="Fz")
        fz_radio.pack(side=tk.LEFT, padx=5)

        def make_changes():
            self.plot_force_data()
            popup.destroy()

        # Button to confirm and close the popup
        confirm_btn = tk.Button(popup, text="Confirm", command=make_changes)
        confirm_btn.pack(pady=10)




if __name__ == "__main__":
    root = tk.Tk()
    app = DisplayApp(root)
    root.mainloop()
