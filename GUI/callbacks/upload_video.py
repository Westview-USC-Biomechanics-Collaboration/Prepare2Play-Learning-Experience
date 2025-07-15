import tkinter as tk
import threading 
from GUI.Timeline import timeline
from PIL import Image, ImageTk
import cv2
from Util.ballDropDetect import ballDropDetect

def uploadVideoCallback(self):
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
        if self.Video.path:
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
    # Open a file dialog for video files
    view_popup = tk.Toplevel(self.master)
    view_popup.title("Select View")

    # Create radio buttons for view options
    tk.Radiobutton(view_popup, text="Long View", variable=self.selected_view, value="Long View").pack(anchor=tk.W)
    tk.Radiobutton(view_popup, text="Top View", variable=self.selected_view, value="Top View").pack(anchor=tk.W)
    tk.Radiobutton(view_popup, text="Short View", variable=self.selected_view, value="Short View").pack(anchor=tk.W)

    # Create a button to confirm the selection
    confirm_button = tk.Button(view_popup, text="Confirm", command=lambda: _upload_video_with_view(self,view_popup))
    confirm_button.pack(pady=10)

    # Block the main window until the popup is closed
    self.master.wait_window(view_popup)

    self.Video.path = tk.filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")])

    # if a video is selected, start a background thread
    def threadTarget():
        process(self)
        self.master.after(0,self._update_video_timeline)
        
    if self.Video.path:
        uploadVideoThread = threading.Thread(target=threadTarget,daemon=True)
        uploadVideoThread.start()


def process(self):
        print(f"[INFO] Video uploaded: {self.Video.path}")
        self.Video.cam = cv2.VideoCapture(self.Video.path)
        self.Video.fps = int(self.Video.cam.get(cv2.CAP_PROP_FPS))
        self.Video.frame_height = int(self.Video.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.Video.frame_width = int(self.Video.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.Video.total_frames = int(self.Video.cam.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.config(to=self.Video.total_frames)   # ---> reconfigure slider value. The max value is the total number of frame in the video
        self.Video.cam.set(cv2.CAP_PROP_POS_FRAMES, self.state.loc)
        self.canvasManager.photo_image1 = self.frameConverter.cvToPillow(camera=self.Video.cam,width=self.Video.frame_width,height=self.Video.frame_height)
        self.canvasID_1 = self.canvasManager.canvas1.create_image(200, 150, image=self.canvasManager.photo_image1, anchor="center")

        # convert timeline image from cvFrame to pillow image
        self.timeline2 = timeline(0, 1)
        videoTimeline = Image.fromarray(self.timeline2.draw_rect(loc=self.state.loc))
        canvas_width = self.video_timeline.winfo_width()
        canvas_height = self.video_timeline.winfo_height()
        videoTimeline = videoTimeline.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        self.timeline_image2 = ImageTk.PhotoImage(videoTimeline)
        self.video_timeline.create_image(0, 0, image=self.timeline_image2, anchor=tk.NW)

        if self.timeline1 is not None:
            # Initialize if not exist
            self.timeline1.update_start_end(0, self.state.force_frame / self.slider['to'])

        # ✅ Offload ballDropDetect to a thread
        def detect_and_finalize():
            print("[INFO] Detecting ball drop...")
            copyCam = cv2.VideoCapture(self.Video.path)
            copyCam.set(cv2.CAP_PROP_POS_FRAMES, 0)
            auto_index = ballDropDetect(copyCam)
            # release memory
            copyCam.release()
            del copyCam
            self.state.loc = auto_index
            print(f"[DEBUG] index for collision is: {auto_index}")
            self.master.after(0, self.label_video)
            self.state.video_loaded = True

        #threading.Thread(target=detect_and_finalize, daemon=True).start()
        self.master.after(0, self.label_video)
        self.state.video_loaded = True