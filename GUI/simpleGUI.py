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
from GUI.callbacks.stepF import stepF
from GUI.callbacks.save import saveCallback
from GUI.callbacks.COM import COMCallback
from GUI.callbacks.vector_overlay_combined import vectorOverlayWithAlignmentCallback
# NEW IMPORTS from modularizing:
from GUI.models.video_state import VideoState
from GUI.models.force_state import ForceState
from GUI.models.state_manager import StateManager
from GUI.layout.canvas_manager import CanvasManager
from GUI.layout.button_manager import ButtonManager
from GUI.layout.label_manager import LabelManager
from GUI.layout.timeline_manager import TimelineManager
from GUI.layout.background_manager import BackgroundManager

class DisplayApp:
    def __init__(self, master):
        self.master = master

        self.Video = VideoState()
        self.Force = ForceState()
        self.state = StateManager()
        self.canvasManager = CanvasManager(self)
        self.buttonManager = ButtonManager(self)
        self.labelManager = LabelManager(self)
        self.timelineManager = TimelineManager(self)
        self.backgroundManager = BackgroundManager(self.master)
        self.background = self.backgroundManager.init_background()  # store background canvas

        # Initialize UI
        self.initUI()

        # lock for multi threading TODO this is not used at all
        self.lock = threading.Lock()     

        # Helper Objects
        self.fileReader = FileFormatter()
        self.COM_helper = COM_helper()  # waited to be initialized  
        self.frameConverter = FrameConverter()   

    def initUI(self):
        self.selected_view = tk.StringVar(value="Long View")
        self.master.title("Multi-Window Display App")
        self.master.geometry("1320x1080")
        self.master.lift()

        self.master.update_idletasks()
        self.initCanvas()
        self.initSlider()
        self.initLabels()
        self.initButtonLayout() 
        self.initTimeline() # init slider must happen before timeline
        self.initSaveWindow()

    def initCanvas(self):
        self.canvasManager.canvas1, self.canvasManager.canvas2, self.canvasManager.canvas3 = self.canvasManager.init_canvases()

        # Add to background
        self.background.create_window(layoutHelper(2, "horizontal"), layoutHelper(2, "vertical"), window=self.canvasManager.canvas1)
        self.background.create_window(layoutHelper(6, "horizontal"), layoutHelper(2, "vertical"), window=self.canvasManager.canvas2)
        self.background.create_window(layoutHelper(10, "horizontal"), layoutHelper(2, "vertical"), window=self.canvasManager.canvas3)


    def initTimeline(self):
        # Place holder for Timeline object
        self.timelineManager.force_canvas, self.timelineManager.video_canvas = self.timelineManager.create_timelines()
        self.background.create_window(700, 700, window=self.timelineManager.force_canvas)
        self.background.create_window(700, 800, window=self.timelineManager.video_canvas)

    
    def initButtonLayout(self):
        self.buttons = self.buttonManager.create_buttons()
        # self.background.create_window(100, 750, window=self.buttons['align'])
        self.background.create_window(650, 350, window=self.buttons['graph_option'])
        self.background.create_window(150, 450, window=self.buttons['step_backward'])
        self.background.create_window(1250, 450, window=self.buttons['step_forward'])
        self.background.create_window(100, 350, window=self.buttons['rotateR'])
        self.background.create_window(280, 350, window=self.buttons['rotateL'])
        self.background.create_window(layoutHelper(3, "horizontal"), 525, window=self.buttons['upload_video'])
        self.background.create_window(layoutHelper(6, "horizontal"), 525, window=self.buttons['upload_force'])
        self.background.create_window(layoutHelper(9, "horizontal"), 525, window=self.buttons['vector_overlay'])
        self.background.create_window(layoutHelper(3, "horizontal"), 575, window=self.buttons['label_video'])
        self.background.create_window(layoutHelper(6, "horizontal"), 575, window=self.buttons['label_force'])
        self.background.create_window(layoutHelper(9, "horizontal"), 575, window=self.buttons['save'])
        self.background.create_window(100, 800, window=self.buttons['Male_COM'])
        self.background.create_window(100, 850, window=self.buttons['Female_COM'])

    def initLabels(self):  
        self.labels = self.labelManager.create_labels()
        self.background.create_window(300,650,window=self.labels['force_timeline'])
        self.background.create_window(300,750,window=self.labels['video_timeline'])

    def initSlider(self):
        self.slider = Scale(self.master, from_=0, to=100, orient="horizontal", label="pick frame", command=self.slider)
        self.background.create_window(700,450,window=self.slider,width=900)
    
    def initSaveWindow(self):
        """Initialize Save window"""
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
        self.COM_intVar = tk.IntVar()
        self.COM_checkbox = None           # check box for COM

    def _on_zoom(self,event,canvas):
        if (canvas == 1):
            if event.delta > 0:
                self.canvasManager.zoom_factor1 *= 1.1  # Zoom in
            else:
                self.canvasManager.zoom_factor1 *= 0.9  # Zoom out

            # Make sure the zoom factor is reasonable
            self.canvasManager.zoom_factor1 = max(0.1, min(self.canvasManager.zoom_factor1, 5.0))  # Limiting zoom range
            print(self.canvasManager.zoom_factor1)
            self.canvasManager.canvas1.delete("all")
            self.canvasManager.photo_image1 = self.frameConverter.cvToPillow(camera=self.Video.cam, width=round(self.Video.frame_width * self.canvasManager.zoom_factor1),
                                                   height=round(self.Video.frame_height * self.canvasManager.zoom_factor1), frame_number=self.state.loc)
            self.canvasManager.canvas1.create_image(self.canvasManager.offset_x1, self.canvasManager.offset_y1, image=self.canvasManager.photo_image1, anchor="center")

        elif (canvas == 3):
            if event.delta > 0:
                self.canvasManagerzoom_factor3 *= 1.1  # Zoom in
            else:
                self.canvasManager.zoom_factor3 *= 0.9  # Zoom out

            # Make sure the zoom factor is reasonable
            self.canvasManager.zoom_factor3 = max(0.1, min(self.canvasManager.zoom_factor3, 5.0))  # Limiting zoom range
            print(self.zoom_factor3)
            self.canvasManager.canvas3.delete("all")
            self.canvasManager.photo_image3 = self.frameConverter.cvToPillow(camera=self.Video.vector_cam, width=round(self.Video.frame_width * self.canvasManager.zoom_factor3),
                                                   height=round(self.Video.frame_height * self.canvasManager.zoom_factor3))
            self.canvasManager.canvas3.create_image(self.canvasManager.offset_x3, self.canvasManager.offset_y3, image=self.canvasManager.photo_image3, anchor="center")

    def _on_drag(self, event, canvas):
        if event.type == "4":  # ButtonPress
            if canvas == 1:
                self.canvasManager.placeloc1 = [event.x, event.y]
            elif canvas == 3:
                self.canvasManager.placeloc3 = [event.x, event.y]

        elif event.type == "6":  # B1-Motion (Dragging)
            if canvas == 1:
                dx = event.x - self.canvasManager.placeloc1[0]
                dy = event.y - self.canvasManager.placeloc1[1]
                self.canvasManager.offset_x1 += dx
                self.canvasManager.offset_y1 += dy
                self.canvasManager.placeloc1 = [event.x, event.y]

                self.canvasManager.canvas1.delete("all")
                self.canvasManager.canvas1.create_image(
                    self.canvasManager.offset_x1,
                    self.canvasManager.offset_y1,
                    image=self.canvasManager.photo_image1,
                    anchor="center"
                )

            elif canvas == 3:
                dx = event.x - self.canvasManager.placeloc3[0]
                dy = event.y - self.canvasManager.placeloc3[1]
                self.canvasManager.offset_x3 += dx
                self.canvasManager.offset_y3 += dy
                self.canvasManager.placeloc3 = [event.x, event.y]

                self.canvasManager.canvas3.delete("all")
                self.canvasManager.canvas3.create_image(
                    self.canvasManager.offset_x3,
                    self.canvasManager.offset_y3,
                    image=self.canvasManager.photo_image3,
                    anchor="center"
                )

        elif event.type == "5":  # ButtonRelease
            if canvas == 1:
                self.canvasManager.canvas1.delete("all")
                self.canvasManager.canvas1.create_image(
                    self.canvasManager.offset_x1,
                    self.canvasManager.offset_y1,
                    image=self.canvasManager.photo_image1,
                    anchor="center"
                )

            elif canvas == 3:
                self.canvasManager.canvas3.delete("all")
                self.canvasManager.canvas3.create_image(
                    self.canvasManager.offset_x3,
                    self.canvasManager.offset_y3,
                    image=self.canvasManager.photo_image3,
                    anchor="center"
                )

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
            # self.master.wait_window(popup)

    def label_video(self):
        self.state.video_align = self.state.loc
        self.timelineManager.timeline2.update_label(self.state.video_align/self.slider['to'])
        self.labels['video_timeline'].config(text=f"Video Timeline (label = {self.state.video_align})")
        self._update_video_timeline()

    def label_force(self):
        self.state.force_align = self.state.loc
        self.timelineManager.timeline1.update_label(self.state.force_align/self.slider['to'])
        self.labels['force_timeline'].config(text=f"Force Timeline (label = {self.state.force_align})")
        self._update_force_timeline()

    # def align(self):
    #     alignCallback(self)

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
        vectorOverlayWithAlignmentCallback(self)

    def startCOM(self):
        COMCallback(self)
    
    def startMaleCOM(self):
        COMCallback(self, "m")
    
    def startFemaleCOM(self):
        COMCallback(self, "f")

    def save(self):
        saveCallback(self)

    def update(self):
        pass

    def on_click(self, event):
        if event.inaxes:  # Check if the click occurred inside the plot area
            print(f"Clicked at: x={event.xdata}, y={event.ydata}")
    
    def plot_force_data(self):
        print("[INFO] plotting force data")
        # Clear previous figure on canvas2
        for widget in self.canvasManager.canvas2.winfo_children():
            widget.destroy()

        canvas_width = self.canvasManager.canvas2.winfo_width()
        canvas_height = self.canvasManager.canvas2.winfo_height()

        # Create a new figure and plot
        self.fig, self.ax = plt.subplots(figsize=(canvas_width / 100, canvas_height / 100), dpi=100)

        # Read data based on plate and force
        plate_number = "1" if self.backgroundManager.plate.get() == "Force Plate 1" else "2"
        x_position = float(self.Force.data.iloc[int(self.state.loc * self.state.step_size + self.state.zoom_pos), 0])
        y_value = float(
            self.Force.data.loc[int(self.state.loc * self.state.step_size + self.state.zoom_pos), f"{self.backgroundManager.option.get()}{plate_number}"])

        # Set x and y
        self.x = self.Force.data.iloc[:, 0]
        self.y = self.Force.data.loc[:, f"{self.backgroundManager.option.get()}{plate_number}"]

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
            0.05, 0.95, f"{self.backgroundManager.plate.get()}\n{self.backgroundManager.option.get()}: {y_value:.2f}",
            transform=self.ax.transAxes, fontsize=12, color='black', verticalalignment='top',
            horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5)
        )

        # Update line and text
        self.zoom_baseline.set_xdata([x_position])
        self.line.set_xdata([x_position])
        self.text_label.set_text(f"{self.backgroundManager.plate.get()}\n{self.backgroundManager.option.get()}: {y_value:.2f}")

        # Embed the Matplotlib figure in the Tkinter canvas
        self.figure_canvas = FigureCanvasTkAgg(self.fig, self.canvasManager.canvas2)
        self.figure_canvas.draw()
        self.figure_canvas.get_tk_widget().place(x=0, y=0, width=canvas_width, height=canvas_height)

        # Enable Matplotlib interactivity
        self.figure_canvas.mpl_connect("button_press_event", self.on_click)  # Example: Connect a click event

        # Optional: Add an interactive toolbar
        toolbar_frame = tk.Frame(self.canvasManager.canvas2)
        toolbar_frame.place(x=0, y=canvas_height - 30, width=canvas_width, height=30)
        toolbar = NavigationToolbar2Tk(self.figure_canvas, toolbar_frame)
        toolbar.update()

        forward = tk.Button(self.canvasManager.canvas2, text="forward", command=lambda: self._plot_move_Button(1))
        self.canvasManager.canvas2.create_window(350, 270, window=forward)

        backward = tk.Button(self.canvasManager.canvas2, text="backward", command=lambda: self._plot_move_Button(-1))
        self.canvasManager.canvas2.create_window(30, 270, window=backward)

        print("[INFO] plot finished")

    def _update_force_timeline(self):
        # Assuming self.timelineManager.timeline1.draw_rect() returns an image
        forceTimeline = Image.fromarray(self.timelineManager.timeline1.draw_rect(loc=self.state.loc / self.slider['to']))

        # Resize the image to fit the canvas size
        canvas_width = self.timelineManager.force_canvas.winfo_width()  # Get the width of the canvas
        canvas_height = self.timelineManager.force_canvas.winfo_height()  # Get the height of the canvas

        # Resize the image to match the canvas size using the new resampling method
        forceTimeline = forceTimeline.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        # Convert the resized image to PhotoImage
        self.timeline_image1 = ImageTk.PhotoImage(forceTimeline)

        # Create the image on the canvas, anchoring it at the top-left (0, 0)
        self.timelineManager.force_canvas.create_image(0, 0, image=self.timeline_image1, anchor=tk.NW)

    def _update_video_timeline(self):
        # Assuming self.timelineManager.video_canvas is the canvas and self.timelineManager.timeline2.draw_rect() returns an image
        videoTimeline = Image.fromarray(self.timelineManager.timeline2.draw_rect(loc=self.state.loc / self.Video.total_frames))

        # Resize the image to fit the canvas size
        canvas_width = self.timelineManager.video_canvas.winfo_width()  # Get the width of the canvas
        canvas_height = self.timelineManager.video_canvas.winfo_height()  # Get the height of the canvas
        # Resize the image to match the canvas size
        videoTimeline = videoTimeline.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        # Convert the resized image to PhotoImage
        self.timeline_image2 = ImageTk.PhotoImage(videoTimeline)

        # Create the image on the canvas, anchoring it at the top-left (0, 0)
        self.timelineManager.video_canvas.create_image(0, 0, image=self.timeline_image2, anchor=tk.NW)

if __name__ == "__main__":
    root = tk.Tk()
    app = DisplayApp(root)
    root.mainloop()