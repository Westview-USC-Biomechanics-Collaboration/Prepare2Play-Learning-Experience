import threading
import os
import pandas as pd
from vector_overlay.vectoroverlay_GUI import VectorOverlay
from GUI.callbacks.ledSyncing import run_led_syncing  
import cv2

def vectorOverlayWithAlignmentCallback(self):
    def threadTarget():
        print("[INFO] Running LED syncing and vector overlay...")
        parent_path = os.path.dirname(self.Video.path)
        video_file = os.path.basename(self.Video.path)
        force_file = os.path.basename(self.Force.path)

        print(f"Name of the video file: {video_file}")
        print(f"Name of the force file: {force_file}")

        # Step 1: Run syncing and get lag
        lag = run_led_syncing(self, parent_path, video_file, force_file)

        with open("lag.txt", "w") as f:
            f.write(str(lag))

        print(f"Detected lag: {lag} frames")

        force_analysis_filename = force_file.replace('.txt', '_Analysis_Force.csv')
        passed_force_file = os.path.join(parent_path, force_analysis_filename)

        print(f"Looking for force file at: {passed_force_file}")

        # Check if file exists
        if not os.path.exists(passed_force_file):
            print(f"Error: Force analysis file not found at {passed_force_file}")
            return

        force_data = pd.read_csv(passed_force_file)

        # Step 3: Run vector overlay with adjusted force data
        self.Video.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
        v = VectorOverlay(data=force_data, video=self.Video.cam)

        temp_video = "vector_overlay_temp.mp4"
        selected = self.selected_view.get()
        if selected == "Long View":
            v.check_corner("Long View")
            v.LongVectorOverlay(outputName=temp_video, lag=lag)
        elif selected == "Short View":
            v.check_corner("Short View")
            v.ShortVectorOverlay(outputName=temp_video, lag=lag)
        elif selected == "Top View":
            v.check_corner("Top View")
            v.TopVectorOverlay(outputName=temp_video, lag=lag)

        self.Video.vector_cam = cv2.VideoCapture(temp_video)
        self.Video.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Step 4: Display overlay result on 3rd canvas
        self.Video.vector_cam.set(cv2.CAP_PROP_POS_FRAMES, self.state.loc)
        self.canvasManager.photo_image3 = self.frameConverter.cvToPillow(camera=self.Video.vector_cam)
        self.canvasManager.canvas3.create_image(200, 150, image=self.canvasManager.photo_image3, anchor="center")

        self.state.vector_overlay_enabled = True

    # Launch in thread
    threading.Thread(target=threadTarget, daemon=True).start()
