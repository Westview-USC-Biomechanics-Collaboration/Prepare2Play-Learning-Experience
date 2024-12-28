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
from datetime import datetime
from io import BytesIO

# our script
from Timeline import timeline
from vector_overlay import vectoroverlay_GUI

"""
I don't have time to clean up the code,
too busy adding new features,
will do if have time...
"""
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
        self._pop_up(text=direction)
        self.selected_view = tk.StringVar(value="Long View") 
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
        self.canvas1.bind("<ButtonPress-1>", lambda event:self._on_drag(event,canvas=1))
        self.canvas1.bind("<B1-Motion>", lambda event:self._on_drag(event,canvas=1))
        self.canvas1.bind("<ButtonRelease-1>", lambda event:self._on_drag(event,canvas=1))
        self.canvas1.bind("<MouseWheel>", lambda event:self._on_zoom(event,canvas=1))

        self.canvas2 = Canvas(self.main_canvas, width=400, height=300, bg="lightgrey")
        self.canvas2.grid(row=0, column=1, padx=20, pady=20)

        self.canvas3 = Canvas(self.main_canvas, width=400, height=300,bg="lightgrey")
        self.canvas3.bind("<ButtonPress-1>", lambda event:self._on_drag(event,canvas=3))
        self.canvas3.bind("<B1-Motion>", lambda event:self._on_drag(event,canvas=3))
        self.canvas3.bind("<ButtonRelease-1>", lambda event:self._on_drag(event,canvas=3))
        self.canvas3.bind("<MouseWheel>", lambda event:self._on_zoom(event,canvas=3))
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

        self.adjustR = tk.Frame(self.main_canvas,bg = "lightgray")
        self.step_forward = tk.Button(self.adjustR, text="+1frame",command=lambda: self._stepF(1))
        self.step_forward.grid(row=0,column=0,padx=20)
        self.rotateR = tk.Button(self.adjustR, text="Rotate clockwise",command=lambda: self._rotateCam(1))
        self.rotateR.grid(row=0,column=1,padx=20)
        self.adjustR.grid(row=2, column=2, pady=10, sticky='ew')


        self.adjustL = tk.Frame(self.main_canvas,bg="lightgray")
        self.step_backward = tk.Button(self.adjustL, text="-1frame", command=lambda: self._stepF(-1))
        self.step_backward.grid(row=0,column=1,padx=20)
        self.rotateL = tk.Button(self.adjustL, text="Rotate counterclockwise",command=lambda: self._rotateCam(-1))
        self.rotateL.grid(row=0,column=0,padx=20)
        self.adjustL.grid(row=2, column=0, padx=20, pady=10, sticky='ew')

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
        self.force_start = None # This variable store the time which user choose to align
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
        self.option = tk.StringVar(value="Fz")

        self.zoom_pos = 0 # force data offset -step size<zoom_pos<+step sized

        # video
        self.rot = 0
        self.video_path = None
        self.cam = None
        self.total_frames = None
        self.vector_cam = None
        self.fps = None
        self.frame_width = None
        self.frame_height = None

        # video canvas 1
        self.zoom_factor1 =1.0
        self.zoom_factor3 = 1.0
        self.placeloc1 = None
        self.placeloc3 = None
        self.offset_x1 = 200
        self.offset_y1 = 150
        self.offset_x3 = 200
        self.offset_y3 = 150

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
        self.save_confirm_button = None    # Final Saving Button
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
            self.photo_image1 = self._display_frame(camera=self.cam, width=round(self.frame_width * self.zoom_factor1),
                                                   height=round(self.frame_height * self.zoom_factor1))
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
            self.photo_image3 = self._display_frame(camera=self.vector_cam, width=round(self.frame_width * self.zoom_factor3),
                                                   height=round(self.frame_height * self.zoom_factor3))
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

    def _pop_up(self, text):
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

    def _openVideo(self, video_path):
        self.cam = cv2.VideoCapture(video_path)
        self.fps = int(self.cam.get(cv2.CAP_PROP_FPS))
        self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.total_frames = int(self.cam.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.config(to=self.total_frames)   # ---> reconfigure slider value. The max value is the total number of frame in the video
        self.cam.set(cv2.CAP_PROP_POS_FRAMES, self.loc)
        self.photo_image1 = self._display_frame(camera=self.cam,width=self.frame_width,height=self.frame_height)
        self.canvasID_1 = self.canvas1.create_image(200, 150, image=self.photo_image1, anchor="center")

    def _display_frame(self,camera,width=400, height=300):
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

    def _rotateCam(self,dir):
        """
        rotate original camera
        """
        self.rot += 90*dir
        name = self.video_path.split("/")[-1][:-4]
        self.loc = 0
        if not self.cam.isOpened():
            print("Error: Could not open camera.")
            return
        self.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
        out = cv2.VideoWriter(f"{name}_rotated{self.rot}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), self.fps,
                              (self.frame_height,self.frame_width))
        while True:
            print(f"{self.cam.get(cv2.CAP_PROP_FRAME_COUNT)}/{self.cam.get(cv2.CAP_PROP_POS_FRAMES)}")
            ret, frame = self.cam.read()
            if not ret:
                break
            rotated_frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE) if dir>0 else cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
            out.write(rotated_frame)
        self.cam.release()
        out.release()
        self.cam = None
        self._openVideo(f"{name}_rotated{self.rot}.mp4")
        print("finish rotating")

    def on_click(self, event):
        if event.inaxes:  # Check if the click occurred inside the plot area
            print(f"Clicked at: x={event.xdata}, y={event.ydata}")
    def _plot_force_data(self):
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
            self.graph_data.loc[int(self.loc * self.step_size + self.zoom_pos), f"{self.option.get()}{plate_number}"])

        # Set x and y
        self.x = self.graph_data.iloc[:, 0]
        self.y = self.graph_data.loc[:, f"{self.option.get()}{plate_number}"]

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

        forward = tk.Button(self.canvas2, text="forward", command=self._forwardButton)
        self.canvas2.create_window(350, 270, window=forward)

        backward = tk.Button(self.canvas2, text="backward", command=self._backwardButton)
        self.canvas2.create_window(30, 270, window=backward)

    """
    The two call function can be written as one with a parameter.
    """
    def _backwardButton(self):
        plate_number = "1" if self.plate.get() == "Force Plate 1" else "2" # plate number
        self.zoom_pos -=1

        # also update the original graph
        x_position = float(self.graph_data.iloc[int(self.loc * self.step_size + self.zoom_pos),0])
        y_value = float(self.graph_data.loc[int(self.loc * self.step_size + self.zoom_pos),f"{self.option.get()}{plate_number}"])
        self.line.set_xdata([x_position])
        self.text_label.set_text(f"{self.plate.get()}\n{self.option.get()}: {y_value:.2f}")
        self.figure_canvas.draw()

        print(self.zoom_pos)
    def _forwardButton(self):
        plate_number = "1" if self.plate.get() == "Force Plate 1" else "2" # plate number
        self.zoom_pos += 1

        x_position = float(self.graph_data.iloc[int(self.loc * self.step_size + self.zoom_pos),0])
        y_value = float(self.graph_data.loc[int(self.loc * self.step_size + self.zoom_pos),f"{self.option.get()}{plate_number}"])
        self.line.set_xdata([x_position])
        self.text_label.set_text(f"{self.plate.get()}\n{self.option.get()}: {y_value:.2f}")
        self.figure_canvas.draw()

        print(self.zoom_pos)

    def _upload_video_with_view(self, popup_window):
        """
        This internal function is called when user clicked 'upload video'
        It gives a pop up window and ask for views and store the answer in self.selected_view
        """
        # Get the selected view
        selected_view = self.selected_view.get()
        print(f"Selected View: {selected_view}") 

        # Close the popup window
        popup_window.destroy()  

        # Proceed with video upload based on the selected view
        #self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")])
        if self.video_path:
            # Handle different views here
            if selected_view == "Long View":                
                self.selected_view = tk.StringVar(value="Long View") 
                print("Long View selected")
            elif selected_view == "Top View":
                # Top View chosen
                self.selected_view = tk.StringVar(value="Top View") 
                print("Top View selected")
            elif selected_view == "Short View":
                # Short view chosen
                self.selected_view = tk.StringVar(value="Short View") 
                print("Short View selected")

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
            self.photo_image1 = self._display_frame(camera=self.cam, width=round(self.frame_width * self.zoom_factor1),
                                                   height=round(self.frame_height * self.zoom_factor1))
            self.canvas1.create_image(self.offset_x1, self.offset_y1, image=self.photo_image1, anchor="center")
            # update video timeline
            self._update_video_timeline()
        if self.vector_cam:
            # draw vector overlay canvas

            self.vector_cam.set(cv2.CAP_PROP_POS_FRAMES, self.loc)
            self.photo_image3 = self._display_frame(camera=self.vector_cam,
                                                   width=round(self.frame_width * self.zoom_factor3),
                                                   height=round(self.frame_height * self.zoom_factor3))
            self.canvas3.create_image(self.offset_x3, self.offset_y3, image=self.photo_image3, anchor="center")

        if self.save_view_canvas:
            # self.save_loc = self.save_scroll_bar.get()
            print(f"You just moved scroll bar to {self.loc}") 
            self.cam.set(cv2.CAP_PROP_POS_FRAMES, self.loc)
            self.save_photoImage = self._display_frame(camera=self.cam)
            self.save_view_canvas.delete("frame_image")
            self.save_view_canvas.create_image(0, 0, image=self.save_photoImage, anchor=tk.NW, tags="frame_image")


        if self.rows is not None:  # somehow self.force_data is not None doesn't work, using self.rows as compensation
            # draw graph canvas
            # normalized_position = int(value) / (self.slider['to'])
            # x_position = self.ax.get_xlim()[0] + normalized_position * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
            try:
                plate_number = "1" if self.plate.get() == "Force Plate 1" else "2"
                x_position = float(self.graph_data.iloc[int(self.loc * self.step_size + self.zoom_pos),0])
                y_value = float(self.graph_data.loc[int(self.loc * self.step_size + self.zoom_pos),f"{self.option.get()}{plate_number}"])
                self.zoom_baseline.set_xdata([self.graph_data.iloc[self.loc*self.step_size,0]])
                self.line.set_xdata([x_position])
                self.text_label.set_text(f"{self.plate.get()}\n{self.option.get()}: {y_value:.2f}")
                self.figure_canvas.draw()

            except IndexError as e:
                print("force data out of range")

            # update force timeline
            self._update_force_timeline()

    def upload_video(self):
        # Open a file dialog for video files
        view_popup = tk.Toplevel(self.master)
        view_popup.title("Select View")

        # Create radio buttons for view options
        tk.Radiobutton(view_popup, text="Long View", variable=self.selected_view, value="Long View").pack(anchor=tk.W)
        tk.Radiobutton(view_popup, text="Top View", variable=self.selected_view, value="Top View").pack(anchor=tk.W)
        tk.Radiobutton(view_popup, text="Short View", variable=self.selected_view, value="Short View").pack(anchor=tk.W)

        # Create a button to confirm the selection
        confirm_button = tk.Button(view_popup, text="Confirm", command=lambda: self._upload_video_with_view(view_popup))
        confirm_button.pack(pady=10)

        # Block the main window until the popup is closed
        self.master.wait_window(view_popup)

        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")])
        if self.video_path:

             
            print(f"Video uploaded: {self.video_path}")
            # display video
            self._openVideo(self.video_path)

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
        def rename_duplicates(lst):
            counts = {}
            new_list = []

            for item in lst:
                if isinstance(item, str) and item != 'abs time (s)':  # Process only strings, skip 'abs time (s)'
                    if item in counts:
                        counts[item] += 1
                        new_list.append(f"{item}{counts[item]}")
                    else:
                        counts[item] = 1
                        new_list.append(f"{item}{counts[item]}")
                else:
                    new_list.append(item)  # Leave non-strings or 'abs time (s)' as is

            return new_list
        # Open a file dialog for any file type
        file_path = filedialog.askopenfilename(title="Select Force Data File",filetypes=[("Excel or CSV Files", "*.xlsx *.xls *.csv")])
        self.force_path = file_path
        print(f"Force data uploaded: {file_path}")
        """
        names = ["abs time (s)", "Fx1", "Fy1", "Fz1", "|Ft1|", "Ax1", "Ay1", "COM px1", "COM py1", "COM pz1",
                 "Fx2", "Fy2", "Fz2", "|Ft2|", "Ax2", "Ay2", "COM px2", "COM py2", "COM pz2"]
        """
        # support both csv and excel
        if file_path.endswith('.xlsx'):
            self.graph_data = pd.read_excel(
                file_path,
            )
        elif file_path.endswith('.csv'):
            self.graph_data = pd.read_csv(
                file_path
            )

        names = rename_duplicates(list(self.graph_data.iloc[16,:]))
        print(names)

        self.graph_data = self.graph_data.iloc[18:,0:len(names)].reset_index(drop=True)
        self.graph_data.columns = names
        self.graph_data = self.graph_data.apply(pd.to_numeric, errors='coerce')

        self.rows = self.graph_data.shape[0]
        try:
            self.step_size = int(600/self.cam.get(cv2.CAP_PROP_FPS)) # rows/frame
        except AttributeError as e:
            print("Video file missing!!!\nProceeding assuming step size is 20 rows/frame")
            self._pop_up("Video file missing!!!\n\nProceeding assuming step size is 20 rows/frame\n\n"
                        "Please reload the force data after uploading the video")
            self.step_size = 20
        self.force_frame = int(self.rows/self.step_size)  # represent num of frames force data can cover

        self._plot_force_data()

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
        """
        12/9 notes:
        When using align for multiple time, the timeline update is not working,
        maybe it's assuming the size of the timeline would not change,
        but we cut data in code, and that's updated in timeline
        """
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
                print(self.graph_data)
            else:
                nan_rows = pd.DataFrame(np.nan, index=range(int(-offset*self.step_size + self.zoom_pos)), columns=self.graph_data.columns)
                self.graph_data = pd.concat([nan_rows, self.graph_data], ignore_index=True)  # We are using + because when we have a positive zoom_pos , the number of added rows is offset*step_size - zoom_pos
                print(self.graph_data)

            print("modify force data")

            # store some output meta data
            self.force_start = self.graph_data.iloc[int(self.video_align*self.step_size),0]

            self.zoom_pos = 0
            self.slider.set(0)
            self.loc = 0
            self._plot_force_data()
        except TypeError as e:
            self._pop_up("Missing label!!!")
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
            self.save_photoImage = self._display_frame(camera=self.vector_cam)
            self.save_view_canvas.delete("frame_image")
            self.save_view_canvas.create_image(0, 0, image=self.save_photoImage, anchor=tk.NW, tags="frame_image")
            
            """
            I notice that when I alter the scroll bar in main window, and then move the scroll bar in toplevel window, the image update
            Possibly because scroll bar in main can also change self.cam. therefore the solution is to link the two together.
            ### solved
            """
        def _export():
            #matplotlib.use('Agg')
            self._pop_up(text="Processing video ...\nThis may take a minute\nClose this window to start saving")
            
            try:
                print(f"\nentry: {self.cushion_entry.get()}\nfps: {self.fps}")
                cushion_frames = int(self.cushion_entry.get()) * (self.fps)
                print(cushion_frames)
            except ValueError as e:
                self._pop_up(text="Invalid cushion time, please put numbers")
                print("invalid input!!\n")
                print(e)
                return
            
            self.cam.set(cv2.CAP_PROP_POS_FRAMES, self.save_start - cushion_frames)
            self.vector_cam.set(cv2.CAP_PROP_POS_FRAMES, self.save_start - cushion_frames)
            count = self.save_start - cushion_frames
            out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps,(self.frame_width, self.frame_height+480))
            print(f"cam1 frame: {self.cam.get(cv2.CAP_PROP_FRAME_COUNT)}\ncam2 frame:{self.vector_cam.get(cv2.CAP_PROP_FRAME_COUNT)}")
            
            # creating matplot graph
            fig1,ax1 = plt.subplots()
            fig2,ax2 = plt.subplots()
            time = self.x

            if(self.selected_view.get()=="Long View"):
                label1_1 = "Fy1"
                label1_2 = "Fz1"
                label2_1 = "Fy2"
                label2_2 = "Fz2"
            elif(self.selected_view.get()=="Short View"):
                label1_1 = "Fx1"
                label1_2 = "Fz1"
                label2_1 = "Fx2"
                label2_2 = "Fz2"
            else: # top view
                label1_1 = "Fy1"
                label1_2 = "Fx1"
                label2_1 = "Fy2"
                label2_2 = "Fx2"
            
            # force plate 1
            y1 = self.graph_data.loc[:,label1_1]
            y2 = self.graph_data.loc[:,label1_2]
            # force plate 2
            y3 = self.graph_data.loc[:,label2_1]
            y4 = self.graph_data.loc[:,label2_2]

            ymax = max(y1.max(),y2.max(),y3.max(),y4.max())
            ymin = min(y1.min(),y2.min(),y3.min(),y4.min())

            self.graph_data.loc[0:self.step_size*self.save_start,:] = np.nan
            self.graph_data.loc[self.step_size*self.save_end:,:] = np.nan

            ax1.clear()
            ax1.set_title(f"Force plate {self.plate.get()} Force Time Graph")
            ax1.set_ylim(ymin, ymax*1.2)
            ax1.plot(time, y1, linestyle='-', color='purple', linewidth=1.5, label="Force horizontal")
            ax1.plot(time, y2, linestyle='-', color='green', linewidth=1.5, label="Force vertical")
            ax1.legend()
            ax1.set_xlabel("Time (s.)")
            ax1.set_ylabel("Forces (N.)")

            line1 = ax1.axvline(x=self.graph_data.iloc[int(count), 0], color='red', linestyle='--', linewidth=1.5)

            ax2.clear()
            ax2.set_title(f"Force plate {self.plate.get()} Force Time Graph")
            ax2.set_ylim(ymin, ymax*1.2)
            ax2.plot(time, y3, linestyle='-', color='orange', linewidth=1.5, label="Force horizontal")
            ax2.plot(time, y4, linestyle='-', color='blue', linewidth=1.5, label="Force vertical")
            ax2.legend()
            ax2.set_xlabel("Time (s.)")
            ax2.set_ylabel("Forces (N.)")

            line2 = ax2.axvline(x=self.graph_data.iloc[int(count), 0], color='red', linestyle='--', linewidth=1.5)
            def render_matplotlib_to_cv2(cur):
                # cur is the row
                LOCtime = self.graph_data.iloc[int(cur),0]
                line1.set_xdata([LOCtime])
                line2.set_xdata([LOCtime])

                # Step 2: Save the plot to a BytesIO object
                buf1 = BytesIO()
                fig1.savefig(buf1, format='png')
                buf1.seek(0)  # Go to the beginning of the BytesIO object

                buf2 = BytesIO()
                fig2.savefig(buf2, format='png')
                buf2.seek(0)

                # Step 3: Convert the BytesIO object to a NumPy array
                image1 = np.asarray(bytearray(buf1.read()), dtype=np.uint8)
                image2 = np.asarray(bytearray(buf2.read()), dtype=np.uint8)

                # Step 4: Decode the byte array to an OpenCV image
                image1 = cv2.imdecode(image1, cv2.IMREAD_COLOR)
                image2 = cv2.imdecode(image2, cv2.IMREAD_COLOR)

                total_width = image1.shape[1] + image2.shape[1]
                if total_width > 1920:
                    raise ValueError("The combined width of image1 and image2 exceeds 1920 pixels.")
                
                gap_width = (1920 - total_width) // 2  # Integer division for the gap width

                gap = np.full((int(image1.shape[0]),gap_width,3),255,dtype=np.uint8)

                return cv2.hconcat([gap,image1,image2,gap])

            # Saving frame with graph
            while(self.vector_cam.isOpened() and count<= self.save_end+cushion_frames):
                ret1, frame1 = self.cam.read()
                ret3, frame3 = self.vector_cam.read()
                if not ret1 or not ret3:
                    # if this calls when the frame_number is equal to the total frame count then the stream has just ended
                    print(f"Can't read frame at position {count}")
                    break
                graphs = render_matplotlib_to_cv2(int(count * self.step_size))  # pass in current row
                if(count<self.save_start):
                    print("doing ori")
                    """
                    12/10 notes
                    combine graphs horizontally and then combine graphs with video vertically
                    export the combined frame, need to test on separate file
                    """
                    combined_frame = cv2.vconcat([frame1,graphs])
                elif(count<=self.save_end):
                    print("doing vector")
                    combined_frame = cv2.vconcat([frame3,graphs])
                else:
                    print("doing ori")
                    combined_frame = cv2.vconcat([frame1, graphs])

                cv2.imshow('Matplotlib Plot', cv2.resize(combined_frame,(960,780)))
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break
                out.write(combined_frame)
                count+=1

            plt.close(fig1)
            plt.close(fig2)
            out.release()
            cv2.destroyAllWindows()
            self._pop_up(text=f"Successfully save vector overlay at {file_path}")
            print(f"Successfully save vector overlay at {file_path}")
            name = file_path.split('/')[-1][:-4]
            with open(f"{file_path[:-4]}.txt","w") as fout:
                fout.write(f"{name}'s metadata\n")
                fout.write(f"Video path: {self.video_path}\n")
                fout.write(f"Total frame: {self.total_frames}\n")
                fout.write(f"FPS: {self.fps}\n")
                fout.write(f"Video start frame: {self.video_align}\n\n")
                
                fout.write(f"Force data path: {self.force_path}\n")
                fout.write(f"Force start frame(before align && with out small adjustments): {self.force_align}\n")
                fout.write(f"Force start time: {self.force_start}\n\n") # using video align because it's position after alignment

                fout.write(f"Cushion time: {self.cushion_entry.get()}\n")
                fout.write(f"Cushion frame: {cushion_frames}\n")  # num of frames before interval of interest and num of frame after if applicable

                fout.write(f"Saving time: {datetime.now()}\n")
                fout.write(f"All rights reserved by Westview PUSD")

            self.master.destroy()


            

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
        self.save_photoImage = self._display_frame(camera=self.vector_cam)
        self.save_view_canvas.create_image(0, 0, image=self.save_photoImage, anchor=tk.NW, tags="frame_image")
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

        
        self.cushion_entry = tk.Entry(self.save_window)
        self.cushion_entry.grid(row=4,column=1,padx=10,pady=10,sticky="w")
        self.cushion_label = tk.Label(self.save_window, text = "set cushion time(s) :")
        self.cushion_label.grid(row=4,column=0,padx=10,pady=10)
        

        self.save_confirm_button = tk.Button(self.save_window,text="export video",command=_export)
        self.save_confirm_button.grid(row=5,column=0,columnspan=3,sticky="nsew",padx=10,pady=10)

        # lift the save window to the front
        self.save_window.lift()

    def vector_overlay(self):
        print("user clicked vector overlay button")
        temp_video = "vector_overlay_temp.mp4"
        self.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
        v = vectoroverlay_GUI.VectorOverlay(data=self.graph_data,video=self.cam)
        
        if(self.selected_view.get()=="Long View"):
            v.LongVectorOverlay(outputName=temp_video)
        elif(self.selected_view.get()=="Short View"):
            v.ShortVectorOverlay(outputName=temp_video)
        elif(self.selected_view.get()=="Top View"):
            v.TopVectorOverlay(outputName=temp_video)

        self.vector_cam = cv2.VideoCapture(temp_video)
        self.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

        """
        display 
        """
        if self.loc>=self.video_align:
            self.vector_cam.set(cv2.CAP_PROP_POS_FRAMES, self.loc - self.video_align)
            self.photo_image3 = self._display_frame(camera=self.vector_cam)
        else:
            self.cam.set(cv2.CAP_PROP_POS_FRAMES, self.loc)
            self.photo_image3 = self._display_frame(camera=self.cam)

    def label_force(self):  # ---> executed when user click label force
        self.force_align = self.loc
        self.timeline1.update_label(self.loc/self.slider['to'])
        self._update_force_timeline()

    def label_video(self):  # ---> executed when user click label video
        self.video_align = self.loc
        self.timeline2.update_label(self.loc/self.slider['to'])
        self._update_video_timeline()

    def graph(self):
        # Create a new popup window
        popup = tk.Toplevel(self.main_canvas)
        popup.title("Force Plate Selection")
        popup.geometry("300x250")

        # Variables to store selected radio button values
        self.plate = tk.StringVar(value="Force Plate 1")
        self.option = tk.StringVar(value="Fx")

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

        fx_radio = tk.Radiobutton(frame2, text="Fx", variable=self.option, value="Fx")
        fx_radio.pack(side=tk.LEFT, padx=5)

        fy_radio = tk.Radiobutton(frame2, text="Fy", variable=self.option, value="Fy")
        fy_radio.pack(side=tk.LEFT, padx=5)

        fz_radio = tk.Radiobutton(frame2, text="Fz", variable=self.option, value="Fz")
        fz_radio.pack(side=tk.LEFT, padx=5)
        
        px_radio = tk.Radiobutton(frame2, text="Ax", variable=self.option, value="Ax")
        px_radio.pack(side=tk.LEFT, padx=5)

        py_radio = tk.Radiobutton(frame2, text="Ay", variable=self.option, value="Ay")
        py_radio.pack(side=tk.LEFT, padx=5)
        def make_changes():
            try:
                if(self.loc>self.force_frame):
                    self.slider.set(0)
                    self.loc = 0

                self._plot_force_data()
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
