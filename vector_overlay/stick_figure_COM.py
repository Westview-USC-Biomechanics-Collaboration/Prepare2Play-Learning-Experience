# --------------------------------------------------------------
#  Modern MediaPipe Pose + Force-Plate Sync (no lag)
# --------------------------------------------------------------
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import multiprocessing as mproc
from multiprocessing import Process, Queue, Pipe
from typing import Optional, Tuple
import time
import traceback
from vector_overlay.Cal_COM import calculateCOM   # <-- keep your COM code

# ------------------------------------------------------------------
# 1. MediaPipe Pose detector (Tasks API – newest)
# ------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def make_landmarker(model_path: str = "pose_landmarker_full.task",
                    num_poses: int = 1,
                    min_detection_confidence: float = 0.5,
                    min_tracking_confidence: float = 0.5):
    """Return a ready-to-use PoseLandmarker (VIDEO mode)."""
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_tracking_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False)
    return PoseLandmarker.create_from_options(options)


# ------------------------------------------------------------------
# 2. Worker – runs MediaPipe on a single frame
# ------------------------------------------------------------------
def pose_worker(frame_q: Queue,
                result_q: Queue,
                model_path: str,
                confidence: float,
                display_com: bool,
                sex: str):
    """One process → one MediaPipe landmarker."""
    landmarker = make_landmarker(model_path=model_path,
                                 min_detection_confidence=confidence,
                                 min_tracking_confidence=confidence)

    while True:
        item = frame_q.get()
        if item is None:                     # sentinel
            result_q.put(None)
            break

        frame_idx, frame, timestamp_ms = item
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = landmarker.detect_for_video(mp_image, int(timestamp_ms))

            # ---- convert to your old dict format ----
            row = {}
            if result.pose_landmarks:
                lmks = result.pose_landmarks[0]          # first (and only) person
                for i, lm in enumerate(lmks):
                    row[f"landmark_{i}_x"] = lm.x
                    row[f"landmark_{i}_y"] = lm.y
                    row[f"landmark_{i}_visibility"] = lm.visibility
            else:
                for i in range(33):
                    row[f"landmark_{i}_x"] = 0.0
                    row[f"landmark_{i}_y"] = 0.0
                    row[f"landmark_{i}_visibility"] = 0.0

            row["frame_index"] = frame_idx

            if display_com and result.pose_landmarks:
                # Build Series exactly like your old code
                xs = [row[f"landmark_{i}_x"] for i in range(33)]
                ys = [row[f"landmark_{i}_y"] for i in range(33)]
                names = [f"{name}_x" for name in
                         ["nose","left_eye_inner","left_eye","left_eye_outer",
                          "right_eye_inner","right_eye","right_eye_outer","left_ear",
                          "right_ear","mouth_left","mouth_right","LSHOULDER",
                          "RSHOULDER","LELBOW","RELBOW","LWRIST","RWRIST",
                          "left_pinky","right_pinky","left_index","right_index",
                          "left_thumb","right_thumb","LHIP","RHIP","LKNEE",
                          "RKNEE","LANKLE","RANKLE","LHEEL","RHEEL","LTOE","RTOE"]]
                series = pd.Series(xs + ys, index=names + [n.replace("_x","_y") for n in names])
                com = calculateCOM(series, sex)
                row["COM_x"] = com[0]
                row["COM_y"] = com[1]

            result_q.put((frame_idx, row, frame.copy()))   # also return frame for drawing
        except Exception as e:
            print(f"[WORKER] error on frame {frame_idx}: {e}\n{traceback.format_exc()}")
            result_q.put((frame_idx, None, None))


# ------------------------------------------------------------------
# 3. Frame reader – adds **exact timestamp** (ms) for perfect sync
# ------------------------------------------------------------------
def frame_reader(frame_q: Queue, video_path: str, fps: float, start_frame: int = 0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[READER] Cannot open video")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    idx = start_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # timestamp in **milliseconds** – MediaPipe VIDEO mode expects this
        timestamp_ms = int(idx / fps * 1000)
        frame_q.put((idx, frame, timestamp_ms))
        idx += 1

    # sentinels
    for _ in range(mp.cpu_count()):
        frame_q.put(None)
    cap.release()

<<<<<<< HEAD

# ------------------------------------------------------------------
# 4. Main processor – one-pass video + CSV + annotated output
# ------------------------------------------------------------------
class PoseProcessor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap.release()
    def SaveToTxt(self,
              sex: str = "male",
              filename: str = "pose_landmarks.csv",
              confidencelevel: float = 0.85,
              displayCOM: bool = True,
              lag_frames: int = 0,
              video_out_path: Optional[str] = None):
    """
    Compatibility method for GUI callbacks.
    Maps old 'SaveToTxt' API → new 'run()' API.
    """
    self.run(csv_path=filename,
             video_out_path=video_out_path,
             sex=sex,
             confidence=confidencelevel,
             display_com=displayCOM,
             lag_frames=lag_frames)
=======
class Processor:
    def __init__(self, video_path):
        #self.cam:cv2.VideoCapture = cam
        self.video_path = video_path
>>>>>>> 6f9942dc3b4259cf438d3220e0c4b1f79eac982e
    
    # ------------------------------------------------------------------
    def run(self,
            csv_path: str = "pose_landmarks.csv",
            video_out_path: Optional[str] = "annotated_video.mp4",
            sex: str = "male",
            confidence: float = 0.85,
            display_com: bool = True,
            lag_frames: int = 0,
            model_path: str = "pose_landmarker_full.task"):
        """
        lag_frames >0  → skip video frames (video starts later)
        lag_frames <0  → skip force data (force starts later) – not used here
        """
        start_frame = max(0, lag_frames)

        # ---- queues ----
        frame_q = Queue(maxsize=60)
        result_q = Queue(maxsize=60)

        # ---- start reader ----
        reader = Process(target=frame_reader,
                         args=(frame_q, self.video_path, self.fps, start_frame))
        reader.start()

        # ---- start workers ----
        num_workers = mproc.cpu_count()
        workers = []
        for i in range(num_workers):
            p = Process(target=pose_worker,
                        args=(frame_q, result_q, model_path,
                              confidence, display_com, sex))
            p.start()
            workers.append(p)

        # ---- video writer (if requested) ----
        writer = None
        if video_out_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_out_path, fourcc, self.fps,
                                     (self.w, self.h))

        # ---- collect results ----
        csv_rows = []
        sentinel_cnt = 0
        processed = 0

        mp_drawing = mp.solutions.drawing_utils
        drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

        while sentinel_cnt < num_workers:
            item = result_q.get()
            if item is None:
                sentinel_cnt += 1
                continue

            frame_idx, row, frame = item
            if row is None:
                continue

            csv_rows.append(row)

            # ---- draw skeleton (optional) ----
            if writer:
                # convert normalized → pixel
                h, w = frame.shape[:2]
                landmarks = []
                for i in range(33):
                    x = int(row[f"landmark_{i}_x"] * w)
                    y = int(row[f"landmark_{i}_y"] * h)
                    vis = row[f"landmark_{i}_visibility"]
                    landmarks.append(mp.tasks.components.containers.Landmark(
                        x=x, y=y, z=0.0, visibility=vis, presence=vis))

                # fake result object for drawing utilities
                fake_result = mp.tasks.vision.PoseLandmarkerResult(
                    pose_landmarks=[landmarks],
                    pose_world_landmarks=None,
                    segmentation_masks=None)

                mp_drawing.draw_landmarks(
                    frame,
                    fake_result.pose_landmarks[0],
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

                # COM circle
                if display_com and "COM_x" in row:
                    cx = int(row["COM_x"] * w)
                    cy = int(row["COM_y"] * h)
                    cv2.circle(frame, (cx, cy), 12, (0, 0, 255), -1)

                writer.write(frame)

            processed += 1
            if processed % 30 == 0:
                print(f"[MAIN] Processed {processed}/{self.total_frames-start_frame} frames")

        # ---- cleanup ----
        reader.join()
        for p in workers:
            p.join()
        if writer:
            writer.release()

        # ---- write CSV ----
        df = pd.DataFrame(csv_rows)
        df = df.sort_values("frame_index").reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        print(f"[DONE] CSV → {csv_path}")
        if video_out_path:
            print(f"[DONE] Video → {video_out_path}")

Processor = PoseProcessor
# ------------------------------------------------------------------
# 5. Example usage (drop-in for your old script)
# ------------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------------
    #  CHANGE THESE TWO LINES ONLY
    # ------------------------------------------------------------------
    video_path = r"C:\Users\Student\Downloads\spu_lr_NS_long_vid01.mov"
    lag_frames = 0                     # <-- read from your lag.txt if you still need it
    # ------------------------------------------------------------------

    # optional: read lag from file (keeps compatibility with your old code)
    try:
        with open("lag.txt", "r") as f:
            lag_frames = int(f.read().strip())
            print(f"[INFO] Using lag = {lag_frames} frames from lag.txt")
    except Exception:
        pass

    processor = PoseProcessor(video_path)
    processor.run(csv_path="pose_landmarks.csv",
                  video_out_path="annotated_video.mp4",
                  sex="male",
                  confidence=0.85,
                  display_com=True,
                  lag_frames=lag_frames)