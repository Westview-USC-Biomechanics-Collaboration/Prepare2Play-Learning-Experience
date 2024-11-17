import sys
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
            "This is the prototype syncing app. Please follow the directions given, otherwise it won't work.\n"
            "REQUIREMENT:\n"
            "a)force is recorded before video recording(force label greater than video label)"
            "\n\n"
            "Step 1: upload video\n"
            "Step 2: upload force data\n"
            "Step 3: click `label video` when slider value is 0\n"
            "Step 4: drag the slider to find the force spike on the graph\n"
            "Step 5: click `label force`\n"
            "Step 6: click align button,\n"
            "Step 7: click `vector overlay` button\n"
            "Step 8: click `save` button and set the output name"
        )
        self.pop_up(text=direction)

        self.master.title("Multi-Window Display App")
        self.master.geometry("1320x1080")
        self.master.lift()

        # Bind the resize event of the master window
        self.master.bind('<Configure>', self.center_canvas)

        # Determine the correct path based on whether the app is running as an exe or not
        if getattr(sys, 'frozen', False):
            # If running from the packaged executable
            app_path = sys._MEIPASS  # Temporary folder where bundled files are extracted
        else:
            # If running from source code
            app_path = os.path.dirname(__file__)

        # scale_factor = 1.0/self.master.tk.call('tk', 'scaling')
        # self.master.tk.call('tk', 'scaling', scale_factor)

        # Load the background image
        img_path = os.path.join(app_path, "lookBack.jpg")
        try:
            image = Image.open(img_path)
            bg_image = ImageTk.PhotoImage(image)
            self.bg_image = bg_image  # Store reference to image here to prevent garbage collection
        except FileNotFoundError:
            print(f"Error: {img_path} not found.")

        # Create a background canvas
        self.background = Canvas(self.master)
        self.background.pack(fill=tk.BOTH, expand=True)

        # Display the background image on the canvas
        self.background.create_image(0, 0, image=self.bg_image, anchor="nw")

        # Adjust the canvas size to fit the image
        self.background.config(width=bg_image.width(), height=bg_image.height())

        # Create a main canvas for content, centered within the background canvas
        self.main_canvas = Canvas(self.background, width=800, height=600, bg="lightgrey")
        self.main_canvas.place(relx=0.5, rely=0.5, anchor="center")

        # Row 0: Create three canvases for display
        self.canvas1 = Canvas(self.main_canvas, width=400, height=300, bg="lightgrey")
        self.canvas1.grid(row=0, column=0, padx=20, pady=20)

        self.canvas2 = Canvas(self.main_canvas, width=400, height=300, bg="lightgrey")
        self.canvas2.grid(row=0, column=1, padx=20, pady=20)

        self.canvas3 = Canvas(self.main_canvas, width=400, height=300, bg="lightgrey")
        self.canvas3.grid(row=0, column=2, padx=20, pady=20)

        # Row 1: Buttons for alignment and graph options
        self.align_button = tk.Button(self.main_canvas, text="Align", command=self.align)
        self.align_button.grid(row=1, column=0, padx=20, pady=10, sticky='ew')

        self.graph_option = tk.Button(self.main_canvas, text="Graphing Options", command=self.graph)
        self.graph_option.grid(row=1, column=2, padx=20, pady=10, sticky='ew')

        # Row 2: Slider to adjust values
        self.slider = Scale(self.main_canvas, from_=0, to=100, orient="horizontal", label="Adjust Value", command=self.update_slider_value)
        self.slider.grid(row=2, column=0, columnspan=3, padx=20, pady=10, sticky='ew')

        # Row 3: Label to display slider value
        self.slider_value_label = Label(self.main_canvas, text="Slider Value: 0")
        self.slider_value_label.grid(row=3, column=0, columnspan=3, padx=20, pady=10, sticky='ew')

        # Row 4: Upload buttons for video and force data
        self.upload_video_button = tk.Button(self.main_canvas, text="Upload Video", command=self.upload_video)
        self.upload_video_button.grid(row=4, column=0, padx=20, pady=10, sticky='ew')

        self.upload_force_button = tk.Button(self.main_canvas, text="Upload Force File", command=self.upload_force_data)
        self.upload_force_button.grid(row=4, column=1, padx=20, pady=10, sticky='ew')

        self.show_vector_overlay = tk.Button(self.main_canvas, text="Vector Overlay", command=self.vector_overlay)
        self.show_vector_overlay.grid(row=4, column=2, padx=20, pady=10, sticky='ew')

        # Row 5: Label buttons for video and force labeling
        self.video_button = tk.Button(self.main_canvas, text="Label Video", command=self.label_video)
        self.video_button.grid(row=5, column=0, padx=20, pady=10, sticky='ew')

        self.force_button = tk.Button(self.main_canvas, text="Label Force", command=self.label_force)
        self.force_button.grid(row=5, column=1, padx=20, pady=10, sticky='ew')

        self.save_button = tk.Button(self.main_canvas, text="Save", command=self.save)
        self.save_button.grid(row=5, column=2, padx=20, pady=10, sticky='ew')

        # Row 6: Force timeline label
        self.force_timeline_label = Label(self.main_canvas, text="Force Timeline (unit = frame)")
        self.force_timeline_label.grid(row=6, column=0, columnspan=3, padx=20, pady=10, sticky='ew')

        # Row 7: Force timeline canvas
        self.force_timeline = Canvas(self.main_canvas, width=1080, height=75, bg="lightblue")
        self.force_timeline.grid(row=7, column=0, columnspan=3, padx=20, pady=10, sticky='ew')

        # Row 8: Video timeline label
        self.video_timeline_label = Label(self.main_canvas, text="Video Timeline (unit = frame)")
        self.video_timeline_label.grid(row=8, column=0, columnspan=3, padx=20, pady=10, sticky='ew')

        # Row 9: Video timeline canvas
        self.video_timeline = Canvas(self.main_canvas, width=1080, height=75, bg="lightblue")
        self.video_timeline.grid(row=9, column=0, columnspan=3, padx=20, pady=10, sticky='ew')

        # Placeholders for images
        self.photo_image1 = None  # Placeholder for image object for canvas1
        self.photo_image2 = None  # Placeholder for image object for canvas2
        self.photo_image3 = None  # Placeholder for image object for canvas3

        # force data
        self.force_path = None
        self.graph_data = None
        self.force_data = None
        self.rows = None
        self.force_frame = None   # convert rows to frames
        self.step_size = None

        # Graph
        self.x = None # x-axis data
        self.y = None # y-axis data
        self.line = None # Initialize the line reference
        self.text_label = None
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

    def center_canvas(self, event=None):
        self.main_canvas.place(x=0, y=0,anchor="center")

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
        # Assuming self.timeline1.draw_rect() returns an image
        forceTimeline = Image.fromarray(self.timeline1.draw_rect(loc=self.loc / self.slider['to']))

        # Resize the image to fit the canvas size
        canvas_width = self.force_timeline.winfo_width()  # Get the width of the canvas
        canvas_height = self.force_timeline.winfo_height()  # Get the height of the canvas

        # Resize the image to match the canvas size using the new resampling method
        forceTimeline = forceTimeline.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        # Convert the resized image to PhotoImage
        self.timeline_image1 = ImageTk.PhotoImage(forceTimeline)

        # Create the image on the canvas, anchoring it at the top-left (0, 0)
        self.force_timeline.create_image(0, 0, image=self.timeline_image1, anchor=tk.NW)

    def update_video_timeline(self):
        # Assuming self.video_timeline is the canvas and self.timeline2.draw_rect() returns an image
        videoTimeline = Image.fromarray(self.timeline2.draw_rect(loc=self.loc / self.total_frames))

        # Resize the image to fit the canvas size
        canvas_width = self.video_timeline.winfo_width()  # Get the width of the canvas
        canvas_height = self.video_timeline.winfo_height()  # Get the height of the canvas
        # Resize the image to match the canvas size
        videoTimeline = videoTimeline.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        # Convert the resized image to PhotoImage
        self.timeline_image2 = ImageTk.PhotoImage(videoTimeline)

        # Create the image on the canvas, anchoring it at the top-left (0, 0)
        self.video_timeline.create_image(0, 0, image=self.timeline_image2, anchor=tk.NW)

    def openVideo(self, video_path):
        self.cam = cv2.VideoCapture(video_path)
        self.total_frames = self.cam.get(cv2.CAP_PROP_FRAME_COUNT)
        self.slider.config(to=self.total_frames)   # ---> reconfigure slider value. The max value is the total number of frame in the video
        self.photo_image1 = self.display_frame(camera=self.cam)
        self.canvas1.create_image(0, 0, image=self.photo_image1, anchor=tk.NW)

    def display_frame(self,camera,vector=False):
        if vector:
            camera.set(cv2.CAP_PROP_POS_FRAMES, self.loc-self.video_align) # pick the corresponding frame to display || the 1st frame is index 0, therefore -1
        else:
            camera.set(cv2.CAP_PROP_POS_FRAMES, self.loc)

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

        canvas_width = self.canvas2.winfo_width()
        canvas_height = self.canvas2.winfo_height()

        # Create a new figure and plot
        self.fig, self.ax = plt.subplots(figsize=(canvas_width/100, canvas_height/100),dpi=100)

        # read data base on plate and force
        plate_number = "1" if self.plate.get() == "Force Plate 1" else "2"
        x_position = float(self.graph_data.iloc[int(self.loc * self.step_size), 0])
        y_value = float(self.graph_data.loc[int(self.loc * self.step_size), f"{self.force.get()}{plate_number}"])

        # set x and y
        self.x = self.graph_data.iloc[:, 0]
        self.y = self.graph_data.loc[:, f"{self.force.get()}{plate_number}"]

        self.ax.plot(self.x, self.y, linestyle='-', color='blue', linewidth = 0.5)
        self.ax.set_title("Force vs. Time")
        self.ax.set_xlabel("Time (s.)")
        self.ax.set_ylabel("Force (N.)")

        # Draw an initial vertical line on the left
        self.line = self.ax.axvline(x=x_position, color='red', linestyle='--', linewidth=1.5)

        # Add a label with the force type inside the plot (top-left corner)
        self.text_label = self.ax.text(0.05, 0.95, f"{self.plate.get()}\n{self.force.get()}: {y_value:.2f}", transform=self.ax.transAxes,
                     fontsize=12, color='black', verticalalignment='top', horizontalalignment='left',
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))

        self.line.set_xdata([x_position])
        self.text_label.set_text(f"{self.plate.get()}\n{self.force.get()}: {y_value:.2f}")

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
    
    second problem, the alignment only works when force label is greater than video label,
    if reversed, the graph will display the last few rows of the force data,
    this can be solved by adding a new column that monitor the global index(frame)
    
    
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
            if self.loc>=self.video_align:
                self.photo_image3 = self.display_frame(camera=self.vector_cam,vector=True)

            else:
                self.photo_image3 = self.display_frame(camera=self.cam)

            self.canvas3.create_image(0, 0, image=self.photo_image3, anchor=tk.NW)

        if self.rows is not None:  # somehow self.force_data is not None doesn't work, using self.rows as compensation
            # draw graph canvas
            # normalized_position = int(value) / (self.slider['to'])
            # x_position = self.ax.get_xlim()[0] + normalized_position * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
            try:
                plate_number = "1" if self.plate.get() == "Force Plate 1" else "2"
                x_position = float(self.graph_data.iloc[int(self.loc * self.step_size),0])
                y_value = float(self.graph_data.loc[int(self.loc * self.step_size),f"{self.force.get()}{plate_number}"])

                self.line.set_xdata([x_position])
                self.text_label.set_text(f"{self.plate.get()}\n{self.force.get()}: {y_value:.2f}")
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
            # Resize the image to fit the canvas size
            canvas_width = self.video_timeline.winfo_width()  # Get the width of the canvas
            canvas_height = self.video_timeline.winfo_height()  # Get the height of the canvas

            # Resize the image to match the canvas size using the new resampling method
            videoTimeline = videoTimeline.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
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

        # make a copy
        self.graph_data = self.force_data.copy()

        self.rows = self.force_data.shape[0]
        try:
            self.step_size = (600/self.cam.get(cv2.CAP_PROP_FPS)) # rows/frame
        except AttributeError as e:
            print("Video file missing!!!\nProceeding assuming step size is 20 rows/frame")
            self.pop_up("Video file missing!!!\n\nProceeding assuming step size is 20 rows/frame\n\n"
                        "Please reload the force data after uploading the video")
            self.step_size = 20
        self.force_frame = int(self.rows/self.step_size)  # represent num of frames force data can cover

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
        # Resize the image to fit the canvas size
        canvas_width = self.force_timeline.winfo_width()  # Get the width of the canvas
        canvas_height = self.force_timeline.winfo_height()  # Get the height of the canvas

        # Resize the image to match the canvas size using the new resampling method
        newforceTimeline = forceTimeline.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        self.timeline_image1 = ImageTk.PhotoImage(newforceTimeline)  # create image object that canvas object accept
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

            self.graph_data = self.graph_data.iloc[int(offset*self.step_size):,:].reset_index(drop=True)
            self.force_data = self.force_data.iloc[int(self.force_align*self.step_size):,:].reset_index(drop=True)
            print("cut force data")

            self.slider.set(0)
            self.loc = 0
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


    def vector_overlay(self):
        print("user clicked vector overlay button")
        temp_video = "vector_overlay_temp.mp4"
        self.cam.set(cv2.CAP_PROP_POS_FRAMES, self.video_align)
        v = vectoroverlay_GUI.VectorOverlay(data=self.force_data,video=self.cam)
        v.LongVectorOverlay(outputName=temp_video)

        self.vector_cam = cv2.VideoCapture(temp_video)
        self.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)


        if self.loc>=self.video_align:
            self.photo_image3 = self.display_frame(camera=self.vector_cam,vector=True)
        else:
            self.photo_image3 = self.display_frame(camera=self.cam)

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
        popup = tk.Toplevel(self.main_canvas)
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
            try:
                self.plot_force_data()
            except AttributeError as e:
                print("Missing force data !!!")
            popup.destroy()

        # Button to confirm and close the popup
        confirm_btn = tk.Button(popup, text="Confirm", command=make_changes)
        confirm_btn.pack(pady=10)




if __name__ == "__main__":
    root = tk.Tk()
    app = DisplayApp(root)
    root.mainloop()
