import sys
import tkinter as tk
from tkinter import filedialog, Canvas, Label, Scale, Frame, Scrollbar, PhotoImage
import cv2
from PIL import Image, ImageTk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
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
        self.bg_image = None
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
        self.canvasID_1 = None
        # Bind mouse events for zoom and drag
        self.canvas1.bind("<ButtonPress-1>", self._on_drag)
        self.canvas1.bind("<B1-Motion>", self._on_drag)
        self.canvas1.bind("<ButtonRelease-1>", self._on_drag)
        self.canvas1.bind("<MouseWheel>", self._on_zoom)

        self.canvas2 = Canvas(self.main_canvas, width=400, height=300, bg="lightgrey")
        self.canvas2.grid(row=0, column=1, padx=20, pady=20)

        self.canvas3 = Canvas(self.main_canvas, width=400, height=300,bg="lightgrey")
        # self.canvas3.create_bitmap(100,100,bitmap="error") # just good to know that there is a bitmap thing
        self.canvas3.grid(row=0, column=2, padx=20, pady=20)

        # Row 1: Buttons for alignment and graph options
        self.align_button = tk.Button(self.main_canvas, text="Align", command=self.align)
        self.align_button.grid(row=1, column=0, padx=20, pady=10, sticky='ew')

        self.graph_option = tk.Button(self.main_canvas, text="Graphing Options", command=self.graph)
        self.graph_option.grid(row=1, column=2, padx=20, pady=10, sticky='ew')

        # Row 2: Slider to adjust values
        self.slider = Scale(self.main_canvas, from_=0, to=100, orient="horizontal", label="pick frame", command=self.update_slider_value)
        self.slider.grid(row=2, column=1, pady=10, sticky='ew')

        self.step_forward = tk.Button(self.main_canvas, text="+1frame",command=lambda: self._stepF(1))
        self.step_forward.grid(row=2, column=2, padx=20, pady=10, sticky='w')

        self.step_backward = tk.Button(self.main_canvas, text="-1frame", command=lambda: self._stepF(-1))
        self.step_backward.grid(row=2, column=0, padx=20, pady=10, sticky='e')

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

        # Zoom Window graphing attributes
        self.zoom_window = None  
        self.canvas_zoom = None
        self.zoom_pos = 0 # force data offset -step size<zoom_pos<+step sized

        # video
        self.video_path = None
        self.cam = None
        self.total_frames = None
        self.vector_cam = None
        self.fps = None
        self.frame_width = None
        self.frame_height = None

        # video canvas 1
        self.zoom_factor =1.0
        self.initplace = None
        self.offset_x = 200
        self.offset_y = 150

        # video zoom window
        self.expand_video = None   # holds a tk button
        self.video_window = None # top level window
        self.big_video = None # canvas object
        self.window_photoImage =None # photoimage place holder

        # timeline
        self.timeline1 = None
        self.timeline2 = None

        # labels
        self.force_align = None
        self.video_align = None

        # saving
        self.save_window = None            # Top level window
        self.save_view_canvas = None       # Canvas
        self.save_photoImage =None         # photoimage for canvas
        self.save_scroll_bar = None        # Scale bar
        self.save_loc = None               # location for sacle bar
        self.save_start_button = None      # Label Start Button
        self.StartLabel = None             # Start label
        self.save_end_button = None        # Label End Button
        self.EndLabel = None               # End Label
        self.save_confrim_button = None    # Final Saving Button
        self.save_start = None             # Start frame
        self.save_end = None               # End frame

        # Global frame/location base on slider
        self.loc = 0

    def center_canvas(self, event=None):
        self.main_canvas.place(x=0, y=0,anchor="center")

    def get_current_frame(self):
        print(self.slider.get())
        return int(self.slider.get()) # return current frame, 1st return 1
    def _stepF(self, dirc):
        if(dirc>0):
            self.loc+=1
        else:
            self.loc-=1
        self.slider.set(self.loc)

    def zoom_frame(self, frame):
        # Resize based on zoom factor
        height, width = frame.shape[:2]
        new_width = int(width * self.zoom_factor)
        new_height = int(height * self.zoom_factor)

        # Resize the frame
        zoomed_frame = cv2.resize(frame, (new_width, new_height))
        return zoomed_frame

    def _on_zoom(self,event):
        self.canvas1.delete("all")
        # Adjust zoom factor based on mouse wheel
        if event.delta > 0:
            self.zoom_factor *= 1.1  # Zoom in
        else:
            self.zoom_factor *= 0.9  # Zoom out

        # Make sure the zoom factor is reasonable
        self.zoom_factor = max(0.1, min(self.zoom_factor, 5.0))  # Limiting zoom range
        print(self.zoom_factor)

        # Update the frame with the new zoom factor
        self.photo_image1 = self.display_frame(camera=self.cam,width=round(self.frame_width*self.zoom_factor),height=round(self.frame_height*self.zoom_factor))
        self.canvas1.create_image(self.offset_x, self.offset_y, image=self.photo_image1, anchor="center")

    def _on_drag(self, event):
        if(event.type=="4"):
            print("initalize place")
            self.initplace = [event.x,event.y]
        if(event.type=="6"):
            self.offset_x += event.x-self.initplace[0]
            self.offset_y += event.y-self.initplace[1]
            self.initplace[0]=event.x
            self.initplace[1]=event.y
            self.canvas1.delete("all")
            self.canvas1.create_image(self.offset_x, self.offset_y, image=self.photo_image1, anchor="center")


        if(event.type=="5"):
            self.canvas1.delete("all")
            self.canvas1.create_image(self.offset_x, self.offset_y, image=self.photo_image1, anchor="center")


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
        self.fps = int(self.cam.get(cv2.CAP_PROP_FPS))
        self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.total_frames = int(self.cam.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.config(to=self.total_frames)   # ---> reconfigure slider value. The max value is the total number of frame in the video
        self.cam.set(cv2.CAP_PROP_POS_FRAMES, self.loc)
        self.photo_image1 = self.display_frame(camera=self.cam,width=self.frame_width,height=self.frame_height)
        self.canvasID_1 = self.canvas1.create_image(200, 150, image=self.photo_image1, anchor="center")
        self.expand_video = tk.Button(self.canvas1, text="Expand", command=self._expand_video)
        self.canvas1.create_window(300, 50, window=self.expand_video)

    def display_frame(self,camera,width=400, height=300):
        """
        This internal function only convert fram to object tkinter accept,
        you need to set the camera frame outside this function
        ex. self.cam.set(...)
        """

        ret, frame = camera.read()  # the `frame` object is now the frame we want

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            frame = Image.fromarray(frame).resize((width, height), resample=Image.BICUBIC) # Resize the frame to 400 * 300
            photoImage = ImageTk.PhotoImage(frame)   # ---> update the image object base on current frame.
            return photoImage

    def on_click(self, event):
        if event.inaxes:  # Check if the click occurred inside the plot area
            print(f"Clicked at: x={event.xdata}, y={event.ydata}")
    def plot_force_data(self):

        # Clear previous figure on canvas2
        for widget in self.canvas2.winfo_children():
            widget.destroy()

        canvas_width = self.canvas2.winfo_width()
        canvas_height = self.canvas2.winfo_height()

        # Create a new figure and plot
        self.fig, self.ax = plt.subplots(figsize=(canvas_width / 100, canvas_height / 100), dpi=100)

        # Read data based on plate and force
        plate_number = "1" if self.plate.get() == "Force Plate 1" else "2"
        x_position = float(self.graph_data.iloc[int(self.loc * self.step_size + self.zoom_pos), 0])
        y_value = float(
            self.graph_data.loc[int(self.loc * self.step_size + self.zoom_pos), f"{self.force.get()}{plate_number}"])

        # Set x and y
        self.x = self.graph_data.iloc[:, 0]
        self.y = self.graph_data.loc[:, f"{self.force.get()}{plate_number}"]

        # Plot data
        self.ax.plot(self.x, self.y, linestyle='-', color='blue', linewidth=0.5)
        self.ax.set_title("Force vs. Time")
        self.ax.set_xlabel("Time (s.)")
        self.ax.set_ylabel("Force (N.)")

        # Draw an initial vertical line on the left
        self.line = self.ax.axvline(x=x_position, color='red', linestyle='--', linewidth=1.5)
        self.zoom_baseline = self.ax.axvline(x=x_position, color='grey', linestyle='--',
                                                 linewidth=1)
        # Add a label with the force type inside the plot (top-left corner)
        self.text_label = self.ax.text(
            0.05, 0.95, f"{self.plate.get()}\n{self.force.get()}: {y_value:.2f}",
            transform=self.ax.transAxes, fontsize=12, color='black', verticalalignment='top',
            horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5)
        )

        # Update line and text
        self.zoom_baseline.set_xdata([x_position])
        self.line.set_xdata([x_position])
        self.text_label.set_text(f"{self.plate.get()}\n{self.force.get()}: {y_value:.2f}")

        # Embed the Matplotlib figure in the Tkinter canvas
        self.figure_canvas = FigureCanvasTkAgg(self.fig, self.canvas2)
        self.figure_canvas.draw()
        self.figure_canvas.get_tk_widget().place(x=0, y=0, width=canvas_width, height=canvas_height)

        # Enable Matplotlib interactivity
        self.figure_canvas.mpl_connect("button_press_event", self.on_click)  # Example: Connect a click event

        # Optional: Add an interactive toolbar
        toolbar_frame = tk.Frame(self.canvas2)
        toolbar_frame.place(x=0, y=canvas_height - 30, width=canvas_width, height=30)
        toolbar = NavigationToolbar2Tk(self.figure_canvas, toolbar_frame)
        toolbar.update()

        # Expand button
        self.expanded_graph = tk.Button(self.canvas2, text="Expand", command=self._expand_graph)
        self.canvas2.create_window(350, 50, window=self.expanded_graph)

        forward = tk.Button(self.canvas2, text="forward", command=self._forwardButton)
        self.canvas2.create_window(350, 270, window=forward)

        backward = tk.Button(self.canvas2, text="backward", command=self._backwardButton)
        self.canvas2.create_window(30, 270, window=backward)



    def _expand_graph(self):
        # Create a new window for the expanded graph
        self.zoom_window = tk.Toplevel(self.master)   # the zoom_window is the main/mast canvas for that small window
        self.zoom_window.title("Expanded Graph")

        # Create a new Matplotlib figure for the zoomed-in data
        self.figzoom, self.axzoom = plt.subplots(figsize=(6, 4))

        # Adjust the data range for the zoomed-in view 
        current_row = int(self.loc*self.step_size) # current row
        plate_number = "1" if self.plate.get() == "Force Plate 1" else "2" # plate number
        zoom_x = self.graph_data.iloc[current_row-self.step_size:current_row+self.step_size,0]  
        zoom_y = self.graph_data.loc[current_row-self.step_size:current_row+self.step_size-1,f"{self.force.get()}{plate_number}"] 

        # debug
        #print(zoom_x,zoom_y)


        self.axzoom.plot(zoom_x, zoom_y,linestyle='-', color='blue', linewidth = 0.5)
        self.axzoom.set_title("Zoomed-in View")
        self.axzoom.set_xlabel("Time")
        self.axzoom.set_ylabel(f"{self.force.get()}")
        self.zoom_textLabel = self.axzoom.text(0.05, 0.95, f"{self.plate.get()}\n{self.force.get()}: {(self.graph_data.loc[current_row+self.zoom_pos,f'{self.force.get()}{plate_number}']):.2f}", transform=self.ax.transAxes,
                     fontsize=12, color='black', verticalalignment='top', horizontalalignment='left',
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
        self.zoom_line = self.axzoom.axvline(x=self.graph_data.iloc[current_row+self.zoom_pos,0], color='red', linestyle='--', linewidth=1.5)
        self.zoom_baseline = self.axzoom.axvline(x=self.graph_data.iloc[current_row,0], color='grey', linestyle='--', linewidth=1)
        # Embed the new plot into the new window
        self.canvas_zoom = FigureCanvasTkAgg(self.figzoom, self.zoom_window)
        self.canvas_zoom.draw()
        self.canvas_zoom.get_tk_widget().pack()

        left_button = tk.Button(self.zoom_window, text="backward", command=self._backwardButton)
        left_button.pack(side="left",padx=20)

        right_button = tk.Button(self.zoom_window, text="forward", command=self._forwardButton)
        right_button.pack(side="right",padx=20)

        self.zoom_window.attributes("-topmost", True)
        self.zoom_window.grab_set()
        self.master.wait_window(self.zoom_window)  # block main window action

    """
    The two call function can be written as one with a parameter.
    """
    def _backwardButton(self):
        plate_number = "1" if self.plate.get() == "Force Plate 1" else "2" # plate number
        self.zoom_pos -=1

        # also update the original graph
        x_position = float(self.graph_data.iloc[int(self.loc * self.step_size + self.zoom_pos),0])
        y_value = float(self.graph_data.loc[int(self.loc * self.step_size + self.zoom_pos),f"{self.force.get()}{plate_number}"])
        self.line.set_xdata([x_position])
        self.text_label.set_text(f"{self.plate.get()}\n{self.force.get()}: {y_value:.2f}")
        self.figure_canvas.draw()

        print(self.zoom_pos)
    def _forwardButton(self):
        plate_number = "1" if self.plate.get() == "Force Plate 1" else "2" # plate number
        self.zoom_pos += 1

        x_position = float(self.graph_data.iloc[int(self.loc * self.step_size + self.zoom_pos),0])
        y_value = float(self.graph_data.loc[int(self.loc * self.step_size + self.zoom_pos),f"{self.force.get()}{plate_number}"])
        self.line.set_xdata([x_position])
        self.text_label.set_text(f"{self.plate.get()}\n{self.force.get()}: {y_value:.2f}")
        self.figure_canvas.draw()

        print(self.zoom_pos)


    def _expand_video(self):
        print("user clicked expand video")
        self.video_window = tk.Toplevel(self.master)
        self.video_window.title("Big video")
        self.big_video = Canvas(self.video_window,width=640, height=480, bg="lightgrey")
        self.big_video.pack(fill="both", expand=True)

        ret, frame = self.cam.read()  # the `frame` object is now the frame we want

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            frame = Image.fromarray(frame).resize((640, 480), resample=Image.BICUBIC)  # Resize the frame to 400 * 300
            self.window_photoImage = ImageTk.PhotoImage(frame)   # ---> update the image object base on current frame.
            self.big_video.create_image(0, 0, image=self.window_photoImage, anchor=tk.NW)

        self.video_window.attributes("-topmost", True)
        #self.video_window.grab_set()



    """
    # methods above are internal functions    
    
    # methods below are buttons and slider that user can interact with
    """
    def update_slider_value(self, value):

        # Update the label with the current slider value
        self.loc = self.get_current_frame()
        self.slider_value_label.config(text=f"Slider Value: {value}")

        # Things that need to be updated when the slider value changes

        if self.cam:
            # draw video canvas
            self.cam.set(cv2.CAP_PROP_POS_FRAMES, self.loc)
            self.photo_image1 = self.display_frame(camera=self.cam, width=round(self.frame_width * self.zoom_factor),
                                                   height=round(self.frame_height * self.zoom_factor))
            self.canvas1.create_image(self.offset_x, self.offset_y, image=self.photo_image1, anchor="center")
            # update video timeline
            self.update_video_timeline()
        if self.vector_cam:
            # draw vector overlay canvas
            if self.loc>=self.video_align:
                self.vector_cam.set(cv2.CAP_PROP_POS_FRAMES, self.loc - self.video_align)
                self.photo_image3 = self.display_frame(camera=self.vector_cam)

            else:
                self.cam.set(cv2.CAP_PROP_POS_FRAMES, self.loc)
                self.photo_image3 = self.display_frame(camera=self.cam)

            self.canvas3.create_image(0, 0, image=self.photo_image3, anchor=tk.NW)
        if self.save_view_canvas:
            # self.save_loc = self.save_scroll_bar.get()
            print(f"You just moved scroll bar to {self.loc}") 
            self.cam.set(cv2.CAP_PROP_POS_FRAMES, self.loc)
            self.save_photoImage = self.display_frame(camera=self.cam)
            self.save_view_canvas.delete("frame_image")
            self.save_view_canvas.create_image(0, 0, image=self.save_photoImage, anchor=tk.NW, tags="frame_image")


        if self.rows is not None:  # somehow self.force_data is not None doesn't work, using self.rows as compensation
            # draw graph canvas
            # normalized_position = int(value) / (self.slider['to'])
            # x_position = self.ax.get_xlim()[0] + normalized_position * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
            try:
                plate_number = "1" if self.plate.get() == "Force Plate 1" else "2"
                x_position = float(self.graph_data.iloc[int(self.loc * self.step_size + self.zoom_pos),0])
                y_value = float(self.graph_data.loc[int(self.loc * self.step_size + self.zoom_pos),f"{self.force.get()}{plate_number}"])
                self.zoom_baseline.set_xdata([self.graph_data.iloc[self.loc*self.step_size,0]])
                self.line.set_xdata([x_position])
                self.text_label.set_text(f"{self.plate.get()}\n{self.force.get()}: {y_value:.2f}")
                self.figure_canvas.draw()

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

            # set step size
            self.step_size = int(600 / self.cam.get(cv2.CAP_PROP_FPS))

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
            self.step_size = int(600/self.cam.get(cv2.CAP_PROP_FPS)) # rows/frame
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

            # debug
            print(offset)
            #check positive or negative offset:
            if(offset>0):
                self.graph_data = self.graph_data.iloc[int(offset*self.step_size + self.zoom_pos):,:].reset_index(drop=True)
            else:
                self.graph_data = self.graph_data.shift(int(-offset*self.step_size + self.zoom_pos))  # We are using + because when we have a positive zoom_pos , the number of added rows is offset*step_size - zoom_pos
                print(self.graph_data)

            self.force_data = self.force_data.iloc[int(self.force_align*self.step_size + self.zoom_pos):,:].reset_index(drop=True)
            print("cut force data")

            self.zoom_pos = 0
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

        def _label(com_in):
            if(com_in<0):# label start
                print("You just labeled start")
                self.save_start = self.save_loc
                self.StartLabel.config(text=f"start frame: {self.save_start}")

            else:
                print("You just labeled end")
                self.save_end = self.save_loc
                self.EndLabel.config(text=f"start frame: {self.save_end}")

        def _scrollBar(value):
            self.save_loc = self.save_scroll_bar.get()
            print(f"You just moved scroll bar to {self.save_loc}") 
            self.vector_cam.set(cv2.CAP_PROP_POS_FRAMES, self.save_loc)
            self.save_photoImage = self.display_frame(camera=self.vector_cam)
            self.save_view_canvas.delete("frame_image")
            self.save_view_canvas.create_image(0, 0, image=self.save_photoImage, anchor=tk.NW, tags="frame_image")
            
            """
            I notice that when I alter the scroll bar in main window, and then move the scroll bar in toplevel window, the image update
            Possibly because scroll bar in main can also change self.cam. therefore the solution is to link the two together.
            ### solved
            """
        def _export():
            self.pop_up(text="Processing video ...")
            self.vector_cam.set(cv2.CAP_PROP_POS_FRAMES, self.save_start)
            count = self.save_start
            out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps,(self.frame_width, self.frame_height))

            while(count<=self.save_end):
                ret, frame = self.vector_cam.read()
                if not ret:
                    # if this calls when the frame_number is equal to the total frame count then the stream has just ended
                    print(f"Can't read frame at position {count}")
                    break

                out.write(frame)
                count+=1
            self.pop_up(text=f"Successfully save vector overlay at {file_path}")
            print(f"Successfully save vector overlay at {file_path}")

        # Creating top level
        self.save_window = tk.Toplevel(self.master)
        self.save_window.title("Save Window")
        self.save_window.geometry("400x560")

        # Freeze the main window
        # self.save_window.grab_set()

        # local variables
        self.save_loc=0

        # layout
        self.save_view_canvas = Canvas(self.save_window,width=400, height=300, bg="lightgrey")
        self.save_view_canvas.create_image(0, 0, image=self.photo_image1, anchor=tk.NW, tags="frame_image")
        self.save_view_canvas.grid(row=0,column=0,columnspan=3,sticky="nsew")

        self.save_scroll_bar = Scale(self.save_window, from_=0, to=self.total_frames, orient="horizontal", label="select start and end", command=_scrollBar)
        self.save_scroll_bar.grid(row=1,column=0,columnspan=3,sticky="nsew",pady=10)

        self.StartLabel = Label(self.save_window,text=f"start frame: {self.save_start}")
        self.StartLabel.grid(row=2,column=0,sticky="nsew",padx=10,pady=10)
        self.save_start_button = tk.Button(self.save_window,text="label start",command=lambda:_label(-1))
        self.save_start_button.grid(row=3,column=0,sticky="nsew",padx=10,pady=10)

        self.EndLabel = Label(self.save_window,text=f"end frame: {self.save_end}")
        self.EndLabel.grid(row=2,column=2,sticky="nsew",padx=10,pady=10)
        self.save_end_button = tk.Button(self.save_window,text="label end",command=lambda:_label(1))
        self.save_end_button.grid(row=3,column=2,sticky="nsew",padx=10,pady=10)

        self.save_confirm_button = tk.Button(self.save_window,text="export video",command=_export)
        self.save_confirm_button.grid(row=4,column=0,columnspan=3,sticky="nsew",padx=10,pady=10)

        self.save_window.lift()

        # shutil.copy("vector_overlay_temp.mp4",file_path)


    def vector_overlay(self):
        print("user clicked vector overlay button")
        temp_video = "vector_overlay_temp.mp4"
        self.cam.set(cv2.CAP_PROP_POS_FRAMES, self.video_align)
        v = vectoroverlay_GUI.VectorOverlay(data=self.force_data,video=self.cam)
        v.LongVectorOverlay(outputName=temp_video)

        self.vector_cam = cv2.VideoCapture(temp_video)
        self.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)


        if self.loc>=self.video_align:
            self.vector_cam.set(cv2.CAP_PROP_POS_FRAMES, self.loc - self.video_align)
            self.photo_image3 = self.display_frame(camera=self.vector_cam)
        else:
            self.cam.set(cv2.CAP_PROP_POS_FRAMES, self.loc)
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
