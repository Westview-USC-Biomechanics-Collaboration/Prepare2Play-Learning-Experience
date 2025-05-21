import sys
import tkinter as tk
from tkinter import filedialog, Canvas, Label, Scale, Frame, Scrollbar, PhotoImage
import cv2
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import math
import threading
from datetime import datetime
from io import BytesIO
import sys

project_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# our script
from vector_overlay.vectoroverlay_GUI import VectorOverlay
from GUI.Timeline import timeline
#from vector_overlay.stick_figure_COM import SaveToTxt  # TODO: this import prevent the exe from running, make it oop
from Util.ballDropDetect import ballDropDetect, forceSpikeDetect
from Util.fileFormater import FileFormatter
from Util.COM_helper import COM_helper
from Util.frameConverter import FrameConverter
from Util.layoutHelper import layoutHelper
from GUI.callbacks.upload_force_data import uploadForceDataCallback
from GUI.callbacks.update_slider_value import sliderCallback
from GUI.callbacks.upload_video import uploadVideoCallback
from GUI.callbacks.align import alignCallback
from GUI.callbacks.graph import graphOptionCallback
from GUI.callbacks.vector_overlay import vectorOverlayCallback
from GUI.callbacks.stepF import stepF

#@dataclass
class Video:
    path: str = None,
    cam : cv2.VideoCapture = None,
    vector_cam: cv2.VideoCapture = None,
    total_frames: int = None,
    frame_width: int = None,
    frame_height: int = None,
    fps: int = None,

#@dataclass
class Force:
    path: str = None,
    data: pd.array = None,
    rows: int = None,

class DisplayApp:
    def __init__(self, master):
        self.master = master

        # Initialize data class
        self.Video = Video()
        self.Force = Force()

        # Initialize UI
        self.initUI()

        # Initialize Global Variables
        self.initGlobalVar()

        # Initialize Global Flags
        self.initGloablFlags()

        # lock for multi threading TODO this is not used at all
        self.lock = threading.Lock()     

        # Helper Objects
        self.fileReader = FileFormatter()
        self.COM_helper = None  # waited to be initialized  
        self.frameConverter = FrameConverter()   

    def initUI(self):
        self.selected_view = tk.StringVar(value="Long View")
        self.master.title("Multi-Window Display App")
        self.master.geometry("1320x1080")
        self.master.lift()

        self.initBackground()
        self.initCanvas()
        self.initSlider()
        self.initLabels()
        self.initButtons()
        self.initTimeline()
    
    def initBackground(self):
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

    def initCanvas(self):
        canvas_width = 400
        # Row 0: Create three canvases for display
        self.canvas1 = Canvas(self.master, width=canvas_width, height=300, bg="lightgrey")
        self.canvasID_1 = None
        # Bind mouse events for zoom and drag
        self.canvas1.bind("<ButtonPress-1>", lambda event:self._on_drag(event,canvas=1))
        self.canvas1.bind("<B1-Motion>", lambda event:self._on_drag(event,canvas=1))
        self.canvas1.bind("<ButtonRelease-1>", lambda event:self._on_drag(event,canvas=1))
        self.canvas1.bind("<MouseWheel>", lambda event:self._on_zoom(event,canvas=1))
        self.canvas1.bind("<Button-4>", lambda event:self._on_zoom_linux(event, canvas=1))  # Linux scroll up
        self.canvas1.bind("<Button-5>", lambda event:self._on_zoom_linux(event, canvas=1))  # Linux scroll down


        self.canvas2 = tk.Canvas(self.master, width=canvas_width, height=300, bg="lightgrey")
        self.canvas2_forward = tk.Button(self.canvas2, text="forward", command=lambda: self._plot_move_Button(1))
        self.canvas2_backward = tk.Button(self.canvas2, text="backward", command=lambda: self._plot_move_Button(-1))
        self.canvas2.create_window(350, 270, window=self.canvas2_forward)
        self.canvas2.create_window(30, 270, window=self.canvas2_backward)

        self.canvas3 = Canvas(self.master, width=canvas_width, height=300,bg="lightgrey")
        self.canvas3.bind("<ButtonPress-1>", lambda event:self._on_drag(event,canvas=3))
        self.canvas3.bind("<B1-Motion>", lambda event:self._on_drag(event,canvas=3))
        self.canvas3.bind("<ButtonRelease-1>", lambda event:self._on_drag(event,canvas=3))
        self.canvas3.bind("<MouseWheel>", lambda event:self._on_zoom(event,canvas=3))
        self.canvas3.bind("<Button-4>", lambda event:self._on_zoom_linux(event, canvas=3))
        self.canvas3.bind("<Button-5>", lambda event:self._on_zoom_linux(event, canvas=3))
        
        self.background.create_window(layoutHelper(2,"horizontal"), layoutHelper(2,"vertical"), window=self.canvas1)  # Place canvas on the background
        self.background.create_window(layoutHelper(6,"horizontal"), layoutHelper(2,"vertical"),window=self.canvas2)
        self.background.create_window(layoutHelper(10,"horizontal"),layoutHelper(2,"vertical"),window=self.canvas3)

        # Placeholders for images
        self.photo_image1 = None  # Placeholder for image object for canvas1
        self.photo_image2 = None  # Placeholder for image object for canvas2
        self.photo_image3 = None  # Placeholder for image object for canvas3

        # video -> canvas 1 & 3
        self.zoom_factor1 =1.0
        self.zoom_factor3 = 1.0
        self.placeloc1 = None
        self.placeloc3 = None
        self.offset_x1 = 200
        self.offset_y1 = 150
        self.offset_x3 = 200
        self.offset_y3 = 150

        # Graph -> Canvas 2
        self.x = None # x-axis data
        self.y = None # y-axis data
        self.line = None # Initialize the line reference
        self.zoom_baseline = None
        self.text_label = None
        self.fig = None
        self.ax = None
        self.figure_canvas = None

        # Graphing options
        self.plate = tk.StringVar(value="Force Plate 1")
        self.option = tk.StringVar(value="Fz")

    def initTimeline(self):
        # Place holder for Timeline object
        self.timeline1:timeline = None
        self.timeline2:timeline = None

        # Timeline Canvas
        self.force_timeline = tk.Canvas(self.master, width=1080, height=75, bg="lightblue")
        self.video_timeline = tk.Canvas(self.master, width=1080, height=75, bg="lightblue")
        
        self.background.create_window(700,700,window=self.force_timeline)
        self.background.create_window(700,800,window=self.video_timeline) 

    def initButtons(self):
        self.align_button = tk.Button(self.master, text="Align", command=self.align)
        self.graph_option = tk.Button(self.master, text="Graphing Options", command=self.graph)
        self.step_forward = tk.Button(self.master, text="+1frame",command=lambda: self.stepF(1))
        self.step_backward = tk.Button(self.master, text="-1frame", command=lambda: self.stepF(-1))
        self.rotateR = tk.Button(self.master, text="Rotate clockwise",command=lambda: self.rotateCam(1))
        self.rotateL = tk.Button(self.master, text="Rotate counterclockwise",command=lambda: self.rotateCam(-1))
        self.upload_video_button = tk.Button(self.master, text="Upload Video", command=self.upload_video)
        self.show_vector_overlay = tk.Button(self.master, text="Vector Overlay", command=self.vector_overlay)
        self.upload_force_button = tk.Button(self.master, text="Upload Force File", command=self.upload_force_data)
        self.video_button = tk.Button(self.master, text="Label Video", command=self.label_video)
        self.force_button = tk.Button(self.master, text="Label Force", command=self.label_force)
        self.save_button = tk.Button(self.master, text="Save", command=self.save)
        self.COM_button = tk.Button(self.master, text="COM", command=self.startCOM)

        self.background.create_window(100,750,window=self.align_button)
        self.background.create_window(650,350,window=self.graph_option)
        self.background.create_window(150,450,window=self.step_backward)
        self.background.create_window(1250,450,window=self.step_forward)
        self.background.create_window(100,350,window=self.rotateR)
        self.background.create_window(280,350,window=self.rotateL)
        self.background.create_window(layoutHelper(3,"horizontal"),525,window=self.upload_video_button)
        self.background.create_window(layoutHelper(6,"horizontal"),525,window=self.upload_force_button)
        self.background.create_window(layoutHelper(9,"horizontal"),525,window=self.show_vector_overlay)
        self.background.create_window(layoutHelper(3,"horizontal"),575,window=self.video_button)
        self.background.create_window(layoutHelper(6,"horizontal"),575,window=self.force_button)
        self.background.create_window(layoutHelper(9,"horizontal"),575,window=self.save_button)
        self.background.create_window(100,800,window=self.COM_button)

    def initLabels(self):
        self.force_timeline_label = Label(self.master, text="Force Timeline (label = frame)")
        self.video_timeline_label = Label(self.master, text="Video Timeline (label = frame)")
        
        self.background.create_window(300,650,window=self.force_timeline_label)
        self.background.create_window(300,750,window=self.video_timeline_label)

    def initSlider(self):
        self.slider = Scale(self.master, from_=0, to=100, orient="horizontal", label="pick frame", command=self.slider)
        self.background.create_window(700,450,window=self.slider,width=900)
    
    def initGlobalVar(self):
        # force data
        self.force_start    = None  # This variable store the time in raw force data which user choose to align
        self.force_frame    = None  # total number of frames could be represented by force data ->calculation: TotalRows/stepsize
        self.step_size      = 10    # step siize unit: rows/frame
        self.zoom_pos       = 0     # canvas 2: force data offset -step size<zoom_pos<+step size
        self.force_align    = None  # Intialize force align value and video align value

        # video
        self.rot = 0 # rotated direction
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
        self.save_confirm_button = None    # Final Saving Button
        self.save_start = None             # Start frame
        self.save_end = None               # End frame

        # Global frame/location base on slider
        self.loc = 0

    def initGloablFlags(self):
        # Global Flags
        self.force_data_flag = False
        self.video_data_flag = False
        self.vector_overlay_flag = False
        self.COM_flag = False

    def _on_zoom(self,event,canvas):
        if (canvas == 1):
            if event.delta > 0:
                self.zoom_factor1 *= 1.1  # Zoom in
            else:
                self.zoom_factor1 *= 0.9  # Zoom out

            # Make sure the zoom factor is reasonable
            self.zoom_factor1 = max(0.1, min(self.zoom_factor1, 5.0))  # Limiting zoom range
            print(self.zoom_factor1)
            self.canvas1.delete("all")
            self.photo_image1 = self.frameConverter.cvToPillow(camera=self.Video.cam, width=round(self.Video.frame_width * self.zoom_factor1),
                                                   height=round(self.Video.frame_height * self.zoom_factor1))
            self.canvas1.create_image(self.offset_x1, self.offset_y1, image=self.photo_image1, anchor="center")

        elif (canvas == 3):
            if event.delta > 0:
                self.zoom_factor3 *= 1.1  # Zoom in
            else:
                self.zoom_factor3 *= 0.9  # Zoom out

            # Make sure the zoom factor is reasonable
            self.zoom_factor3 = max(0.1, min(self.zoom_factor3, 5.0))  # Limiting zoom range
            print(self.zoom_factor3)
            self.canvas3.delete("all")
            self.photo_image3 = self.frameConverter.cvToPillow(camera=self.Video.vector_cam, width=round(self.Video.frame_width * self.zoom_factor3),
                                                   height=round(self.Video.frame_height * self.zoom_factor3))
            self.canvas3.create_image(self.offset_x3, self.offset_y3, image=self.photo_image3, anchor="center")

    def _on_drag(self, event, canvas):
        if(event.type=="4"):
            if(canvas==1):
                self.placeloc1 = [event.x,event.y]
            elif(canvas==3):
                self.placeloc3 = [event.x,event.y]
        if(event.type=="6"):
            if(canvas==1):
                self.offset_x1 += event.x - self.placeloc1[0]
                self.offset_y1 += event.y - self.placeloc1[1]
                self.placeloc1[0] = event.x
                self.placeloc1[1] = event.y
                self.canvas1.delete("all")
                self.canvas1.create_image(self.offset_x1, self.offset_y1, image=self.photo_image1, anchor="center")
            elif(canvas==3):
                print(event.x,event.y)
                self.offset_x3 += event.x - self.placeloc3[0]
                self.offset_y3 += event.y - self.placeloc3[1]
                self.placeloc3[0] = event.x
                self.placeloc3[1] = event.y
                self.canvas3.delete("all")
                self.canvas3.create_image(self.offset_x3, self.offset_y3, image=self.photo_image3, anchor="center")

        if(event.type=="5"):
            if(canvas==1):
                self.canvas1.delete("all")
                self.canvas1.create_image(self.offset_x1, self.offset_y1, image=self.photo_image1, anchor="center")
            elif(canvas==3):
                self.canvas3.delete("all")
                self.canvas3.create_image(self.offset_x3, self.offset_y3, image=self.photo_image3, anchor="center")

    def _pop_up(self, text, follow=False):
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

        if(not follow):
            # Make the popup modal (blocks interaction with the main window)
            popup.grab_set()
            # Wait for the popup to be destroyed before returning to the main window
            self.master.wait_window(popup)

    def label_video(self):
        self.video_align = self.loc
        self.timeline2.update_label(self.video_align/self.slider['to'])
        self.video_timeline_label.config(text=f"Video Timeline (label = {self.video_align})")
        self._update_video_timeline()

    def label_force(self):
        self.force_align = self.loc
        self.timeline1.update_label(self.force_align/self.slider['to'])
        self.force_timeline_label.config(text=f"Force Timeline (label = {self.force_align})")
        self._update_force_timeline()

    def align(self):
        alignCallback(self)

    def graph(self):
        graphOptionCallback(self)
    
    def slider(self,value):
        sliderCallback(self,value)

    def stepF(self,num:int):
        stepF(self,num)

    def rotateCam(self,num:int):
        pass

    def upload_video(self):
        uploadVideoCallback(self)

    def upload_force_data(self):
        uploadForceDataCallback(self)

    def vector_overlay(self):
        vectorOverlayCallback(self)

    def startCOM(self):
        pass

    def save(self):
        pass

    def update(self):
        pass

    def on_click(self, event):
        if event.inaxes:  # Check if the click occurred inside the plot area
            print(f"Clicked at: x={event.xdata}, y={event.ydata}")
    
    def plot_force_data(self):
        print("[INFO] plotting force data")
        # Clear previous figure on canvas2
        for widget in self.canvas2.winfo_children():
            widget.destroy()

        canvas_width = self.canvas2.winfo_width()
        canvas_height = self.canvas2.winfo_height()

        # Create a new figure and plot
        self.fig, self.ax = plt.subplots(figsize=(canvas_width / 100, canvas_height / 100), dpi=100)

        # Read data based on plate and force
        plate_number = "1" if self.plate.get() == "Force Plate 1" else "2"
        x_position = float(self.Force.data.iloc[int(self.loc * self.step_size + self.zoom_pos), 0])
        y_value = float(
            self.Force.data.loc[int(self.loc * self.step_size + self.zoom_pos), f"{self.option.get()}{plate_number}"])

        # Set x and y
        self.x = self.Force.data.iloc[:, 0]
        self.y = self.Force.data.loc[:, f"{self.option.get()}{plate_number}"]

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
            0.05, 0.95, f"{self.plate.get()}\n{self.option.get()}: {y_value:.2f}",
            transform=self.ax.transAxes, fontsize=12, color='black', verticalalignment='top',
            horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5)
        )

        # Update line and text
        self.zoom_baseline.set_xdata([x_position])
        self.line.set_xdata([x_position])
        self.text_label.set_text(f"{self.plate.get()}\n{self.option.get()}: {y_value:.2f}")

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

        forward = tk.Button(self.canvas2, text="forward", command=lambda: self._plot_move_Button(1))
        self.canvas2.create_window(350, 270, window=forward)

        backward = tk.Button(self.canvas2, text="backward", command=lambda: self._plot_move_Button(-1))
        self.canvas2.create_window(30, 270, window=backward)

        print("[INFO] plot finished")

    def _update_force_timeline(self):
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

    def _update_video_timeline(self):
        # Assuming self.video_timeline is the canvas and self.timeline2.draw_rect() returns an image
        videoTimeline = Image.fromarray(self.timeline2.draw_rect(loc=self.loc / self.Video.total_frames))

        # Resize the image to fit the canvas size
        canvas_width = self.video_timeline.winfo_width()  # Get the width of the canvas
        canvas_height = self.video_timeline.winfo_height()  # Get the height of the canvas
        # Resize the image to match the canvas size
        videoTimeline = videoTimeline.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        # Convert the resized image to PhotoImage
        self.timeline_image2 = ImageTk.PhotoImage(videoTimeline)

        # Create the image on the canvas, anchoring it at the top-left (0, 0)
        self.video_timeline.create_image(0, 0, image=self.timeline_image2, anchor=tk.NW)

if __name__ == "__main__":
    root = tk.Tk()
    app = DisplayApp(root)
    root.mainloop()