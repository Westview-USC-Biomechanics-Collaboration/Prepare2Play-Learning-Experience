import threading
import os
from vector_overlay.vectoroverlay_GUI import VectorOverlay
from newData.ledSyncing import run_led_syncing  # rename or move if needed
import cv2

def vectorOverlayWithAlignmentCallback(self):
    def threadTarget():
        print("[INFO] Running LED syncing and vector overlay...")
        parent_path = os.path.dirname(self.Video.path)
        video_file = os.path.basename(self.Video.path)
        force_file = os.path.basename(self.Force.path)

        # Step 1: Run syncing and get lag
        lag = run_led_syncing(parent_path, video_file, force_file)

        # Step 2: Apply lag by trimming force data
        if lag > 0:
            trim_rows = lag * self.state.step_size
            self.Force.data = self.Force.data.iloc[trim_rows:].reset_index(drop=True)

        # Step 3: Run vector overlay with adjusted force data
        self.Video.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
        v = VectorOverlay(data=self.Force.data, video=self.Video.cam)

        temp_video = "vector_overlay_temp.mp4"
        selected = self.selected_view.get()
        if selected == "Long View":
            v.LongVectorOverlay(outputName=temp_video, lag=lag)
        elif selected == "Short View":
            v.ShortVectorOverlay(outputName=temp_video, lag=lag)
        elif selected == "Top View":
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
