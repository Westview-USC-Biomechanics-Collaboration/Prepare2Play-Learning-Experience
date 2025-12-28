import traceback
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from vector_overlay.Cal_COM import calculateCOM
import csv
import time
import threading
import multiprocessing as processing
from multiprocessing import Process, Queue, Pipe
import queue
import cv2 as cv
from vector_overlay.select_corners import select_points
import os
from Util.COM_helper import COM_helper


"""
Fixed COM_vectoroverlay.py - Only processes aligned frame boundaries

Key changes:
1. Only reads frames that exist in df_aligned (not all frames)
2. Fixed multiprocessing to avoid TensorFlow import issues
3. Added proper frame range calculation based on alignment
"""

import traceback
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from vector_overlay.Cal_COM import calculateCOM
import time
import threading
import multiprocessing as processing
from multiprocessing import Process, Queue
import queue
import cv2 as cv
from vector_overlay.select_corners import select_points
from Util.COM_helper import COM_helper


def process_frame(q: processing.Queue, resultsVector_queue: processing.Queue, resultsCOM_queue: processing.Queue, cfg):
    """
    Worker process for COM calculation and vector overlay.
    
    IMPORTANT: Import mediapipe HERE, not at module level, to avoid
    Windows multiprocessing issues with TensorFlow DLL loading.
    """
    import mediapipe as mp
    import cv2
    import pandas as pd
    from vector_overlay.Cal_COM import calculateCOM
    
    sex = cfg['sex']
    confidencelevel = cfg['confidence']
    displayCOM = cfg['displayCOM']

    def rect_to_trapezoid(x, y, rect_width, rect_height, trapezoid_coords, short=False):
        x = np.clip(x, 0, rect_width)
        y = np.clip(y, 0, rect_height)
        (tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y) = trapezoid_coords
        left_x = bl_x + (tl_x - bl_x) * y
        right_x = br_x + (tr_x - br_x) * y
        trapezoid_width = right_x - left_x
        new_x = left_x + x * trapezoid_width
        top_y = (tl_y + tr_y) / 2
        bottom_y = (bl_y + br_y) / 2
        if short:
            new_y = bottom_y + y * (top_y - bottom_y)
        else:
            new_y = top_y + y * (bottom_y - top_y)
        return (int(new_x), int(new_y))

    def drawArrows(frame, xf1, xf2, yf1, yf2, px1, px2, py1, py2, corners, short=False):
        if short:
            point_pair1 = rect_to_trapezoid(px1, py1, 1, 1,
                                           [corners[0], corners[1], corners[2], corners[3]], short=True)
            point_pair2 = rect_to_trapezoid(px2, py2, 1, 1,
                                           [corners[4], corners[5], corners[6], corners[7]], short=True)
        else:
            point_pair1 = rect_to_trapezoid(px1, py1, 1, 1,
                                           [corners[0], corners[1], corners[2], corners[3]])
            point_pair2 = rect_to_trapezoid(px2, py2, 1, 1,
                                           [corners[4], corners[5], corners[6], corners[7]])

        end_point_1 = (int(point_pair1[0] + xf1), int(point_pair1[1] - yf1))
        end_point_2 = (int(point_pair2[0] + xf2), int(point_pair2[1] - yf2))
        cv.arrowedLine(frame, point_pair1, end_point_1, (0, 255, 0), 4)
        cv.arrowedLine(frame, point_pair2, end_point_2, (255, 0, 0), 4)

    pose_landmark_names = {
        0: "nose", 11: "LSHOULDER", 12: "RSHOULDER", 13: "LELBOW", 14: "RELBOW",
        15: "LWRIST", 16: "RWRIST", 17: "LPINKY", 18: "left_index", 19: "LTHUMB",
        20: "RPINKY", 21: "right_index", 22: "RTHUMB", 23: "LHIP", 24: "RHIP", 25: "LKNEE",
        26: "RKNEE", 27: "LANKLE", 28: "RANKLE", 29: "LHEEL", 30: "RHEEL",
        31: "LTOE", 32: "RTOE"
    }

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=confidencelevel, model_complexity=2) as pose:
        print(f"[PROCESS {processing.current_process().name}] Started.")
        
        while True:
            try:
                item = q.get(timeout=10)
            except queue.Empty:
                print(f"[PROCESS {processing.current_process().name}] Timed out waiting for frame.")
                resultsCOM_queue.put(None)
                break

            if item is None:
                print(f"[PROCESS {processing.current_process().name}] Got sentinel, exiting.")
                resultsCOM_queue.put(None)
                resultsVector_queue.put(None)
                break

            frame_index, frame_data = item
            
            if frame_data is None:
                print(f"[PROCESS {processing.current_process().name}] None frame at {frame_index}.")
                resultsCOM_queue.put(None)
                resultsVector_queue.put((frame_index, frame_data))
                continue
            
            try:
                full_frame = frame_data
                frame = cv2.resize(frame_data, (0, 0), fx=0.3, fy=0.3)
                
                results = pose.process(frame)
                if results.pose_landmarks:
                    pose_landmarks_list = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]
                else:
                    pose_landmarks_list = [[0.0, 0.0, 0.0]] * 33

                # Create row with landmark data
                row = {"frame_index": frame_index}
                for i in range(33):
                    row[f"landmark_{i}_x"] = pose_landmarks_list[i][0]
                    row[f"landmark_{i}_y"] = pose_landmarks_list[i][1]
                    row[f"landmark_{i}_visibility"] = 1.0 if any(pose_landmarks_list[i]) else 0.0

                if displayCOM:
                    datain = pd.Series(
                        [row[f"landmark_{i}_x"] for i in range(33)] + [row[f"landmark_{i}_y"] for i in range(33)],
                        index=[f"{pose_landmark_names.get(i, f'landmark_{i}')}_x" for i in range(33)] +
                              [f"{pose_landmark_names.get(i, f'landmark_{i}')}_y" for i in range(33)]
                    )
                    dataout = calculateCOM(datain, sex)
                    row["COM_x"] = dataout[0]
                    row["COM_y"] = dataout[1]

                resultsCOM_queue.put(row)

                # Get force data for this frame
                f_idx = frame_index
                if 0 <= f_idx < len(cfg['fx1']):
                    _fx1 = -cfg['fy1'][f_idx]
                    _fx2 = -cfg['fy2'][f_idx]
                    _fy1 = cfg['fz1'][f_idx]
                    _fy2 = cfg['fz2'][f_idx]
                    _px1 = cfg['py1'][f_idx]
                    _py1 = cfg['px1'][f_idx]
                    _px2 = cfg['py2'][f_idx]
                    _py2 = cfg['px2'][f_idx]
                    
                    drawArrows(full_frame, _fx1, _fx2, _fy1, _fy2, _px1, _px2, _py1, _py2, cfg['corners'])

                resultsVector_queue.put((frame_index, full_frame))

            except Exception as e:
                print(f"Error processing frame {frame_index}: {e}")
                traceback.print_exc()
                resultsVector_queue.put((frame_index, full_frame))
                resultsCOM_queue.put(None)


def frame_reader(frame_queue: processing.Queue, video_path, frame_indices, num_workers):
    """
    Read ONLY the frames that exist in df_aligned.
    
    Args:
        frame_queue: Queue to put frames into
        video_path: Path to video
        frame_indices: List of frame indices from df_aligned['FrameNumber']
        num_workers: Number of worker processes
    """
    print(f"[READER] Opening video: '{video_path}'")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[READER] ERROR: Could not open video")
        for _ in range(num_workers):
            frame_queue.put(None)
        return
    
    # Sort frame indices to read sequentially
    sorted_indices = sorted(frame_indices)
    
    print(f"[INFO] Reading frames {sorted_indices[0]} to {sorted_indices[-1]}...")
    print(f"[INFO] Total frames to process: {len(sorted_indices)}")
    
    frames_read = 0
    for frame_idx in sorted_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"[READER] Warning: Could not read frame {frame_idx}")
            continue
        
        try:
            frame_queue.put((frame_idx, frame.copy()))
            frames_read += 1
            
            if frames_read % 100 == 0:
                print(f"[READER] Loaded {frames_read} frames...")
                
        except Exception as e:
            print(f"[READER] Error putting frame {frame_idx}: {e}")
            break
    
    cap.release()
    print(f"[READER] Done reading {frames_read} frames. Sending sentinels...")
    
    for _ in range(num_workers):
        frame_queue.put(None)
    
    print("[READER] Sent all sentinels.")


def writer_process(results_q, out_path, fps, frame_size, num_workers):
    """Write processed frames to video file in order."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, frame_size)
    
    next_to_write = 0
    buffer = {}
    sentinels = 0
    
    while True:
        if sentinels >= num_workers and not buffer:
            break
        
        try:
            item = results_q.get(timeout=5.0)
        except queue.Empty:
            continue
        
        if item is None:
            sentinels += 1
            continue
        
        idx, frame = item
        buffer[idx] = frame
        
        while next_to_write in buffer:
            out.write(buffer.pop(next_to_write))
            next_to_write += 1
    
    out.release()
    print(f"[WRITER] Finished writing video")


class Processor:
    def __init__(self, video_path, data_df, lag, output_mp4, force_fps=None):
        """
        Initialize processor with aligned dataframe.
        
        Args:
            video_path: Path to video file
            data_df: df_aligned - contains FrameNumber and force data
            lag: Alignment lag (frames)
            output_mp4: Output video path
            force_fps: Force sampling rate (optional)
        """
        self.video_path = video_path
        self.data = data_df  # This is df_aligned!
        self.lag = lag
        self.output_mp4 = output_mp4
        self.force_fps = force_fps

        print(f"[Processor.__init__] Video: {video_path}")
        print(f"[Processor.__init__] Aligned data shape: {data_df.shape}")
        print(f"[Processor.__init__] Lag: {lag} frames")

        # Open video for metadata
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
        
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f"[Processor.__init__] Video: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
        print(f"[Processor.__init__] Total video frames: {self.frame_count}")

        self.corners = []
        
        # Extract force vectors from aligned data
        self.readData()

    def check_corner(self, view):
        """Detect force plate corners in video."""
        cap = cv2.VideoCapture(self.video_path)
        self.corners = select_points(self, cap=cap, view=view)
        cap.release()

    def readData(self):
        """
        Extract force data arrays from df_aligned.
        
        IMPORTANT: df_aligned already has proper alignment, so we just
        extract the force columns into arrays for each frame.
        """
        print("\n[Processor.readData] Extracting force data from aligned dataframe...")
        
        # Use the aligned dataframe directly
        df = self.data.copy()
        
        # Check what columns we have
        print(f"[readData] Available columns: {list(df.columns)}")
        
        # Map column names (handle both formats)
        force_cols = {}
        for old_col, new_col in [
            ('FP1_Fx', 'Fx1'), ('FP1_Fy', 'Fy1'), ('FP1_Fz', 'Fz1'),
            ('FP1_Ax', 'Ax1'), ('FP1_Ay', 'Ay1'),
            ('FP2_Fx', 'Fx2'), ('FP2_Fy', 'Fy2'), ('FP2_Fz', 'Fz2'),
            ('FP2_Ax', 'Ax2'), ('FP2_Ay', 'Ay2')
        ]:
            if old_col in df.columns:
                force_cols[new_col] = old_col
            elif new_col in df.columns:
                force_cols[new_col] = new_col
        
        # Extract force arrays
        self.fx1 = df[force_cols.get('Fx1', 'Fx1')].fillna(0.0).to_numpy()
        self.fy1 = df[force_cols.get('Fy1', 'Fy1')].fillna(0.0).to_numpy()
        self.fz1 = df[force_cols.get('Fz1', 'Fz1')].fillna(0.0).to_numpy()
        
        self.fx2 = df[force_cols.get('Fx2', 'Fx2')].fillna(0.0).to_numpy()
        self.fy2 = df[force_cols.get('Fy2', 'Fy2')].fillna(0.0).to_numpy()
        self.fz2 = df[force_cols.get('Fz2', 'Fz2')].fillna(0.0).to_numpy()
        
        # Extract pressure positions
        ax1 = df[force_cols.get('Ax1', 'Ax1')].fillna(0.0).to_numpy()
        ay1 = df[force_cols.get('Ay1', 'Ay1')].fillna(0.0).to_numpy()
        ax2 = df[force_cols.get('Ax2', 'Ax2')].fillna(0.0).to_numpy()
        ay2 = df[force_cols.get('Ay2', 'Ay2')].fillna(0.0).to_numpy()
        
        # Normalize pressure coordinates
        self.px1 = np.clip((ax1 + 0.3) / 0.6, 0, 1)
        self.py1 = np.clip((ay1 + 0.45) / 0.9, 0, 1)
        self.px2 = np.clip((ax2 + 0.3) / 0.6, 0, 1)
        self.py2 = np.clip((ay2 + 0.45) / 0.9, 0, 1)
        
        # Normalize forces for visualization
        max_force = max(
            np.max(np.abs(self.fx1)), np.max(np.abs(self.fx2)),
            np.max(np.abs(self.fy1)), np.max(np.abs(self.fy2)),
            np.max(np.abs(self.fz1)), np.max(np.abs(self.fz2))
        )
        
        if max_force > 0:
            scale_factor = min(self.frame_height, self.frame_width) * 0.8 / max_force
            self.fx1 *= scale_factor
            self.fy1 *= scale_factor
            self.fz1 *= scale_factor
            self.fx2 *= scale_factor
            self.fy2 *= scale_factor
            self.fz2 *= scale_factor
        
        print(f"[readData] Extracted {len(self.fx1)} force samples from aligned data")

    def SaveToTxt(self, sex, filename, confidencelevel=0.85, displayCOM=False):
        """
        Process video to extract COM landmarks.
        
        ONLY processes frames that exist in df_aligned (the aligned subset).
        """
        startTime = time.time()

        # Get frame indices from aligned data
        frame_indices = self.data['FrameNumber'].dropna().astype(int).tolist()
        
        print(f"\n[SaveToTxt] Processing {len(frame_indices)} aligned frames")
        print(f"[SaveToTxt] Frame range: {min(frame_indices)} to {max(frame_indices)}")

        cfg = dict(
            sex=sex, confidence=confidencelevel, displayCOM=displayCOM,
            fps=self.fps, frame_count=self.frame_count,
            frame_width=self.frame_width, frame_height=self.frame_height,
            lag=self.lag, corners=self.corners,
            fx1=self.fx1, fx2=self.fx2, fy1=self.fy1, fy2=self.fy2,
            fz1=self.fz1, fz2=self.fz2, px1=self.px1, px2=self.px2,
            py1=self.py1, py2=self.py2
        )

        frame_queue = Queue(maxsize=200)
        resultCOM_queue = Queue()
        resultVector_queue = Queue()

        num_workers = min(6, processing.cpu_count())

        # Start reader with ONLY aligned frame indices
        reader_thread = threading.Thread(
            target=frame_reader,
            args=(frame_queue, self.video_path, frame_indices, num_workers)
        )
        reader_thread.start()

        # Start writer
        writer = processing.Process(
            target=writer_process,
            args=(resultVector_queue, self.output_mp4, self.fps,
                  (self.frame_width, self.frame_height), num_workers),
            daemon=True
        )
        writer.start()

        # Start workers
        workers = []
        for i in range(num_workers):
            p = processing.Process(
                target=process_frame,
                args=(frame_queue, resultVector_queue, resultCOM_queue, cfg),
                daemon=True
            )
            p.start()
            workers.append(p)

        reader_thread.join()
        print("[SaveToTxt] Reader thread finished")

        # Collect COM results
        output = []
        sentinel_count = 0
        while sentinel_count < num_workers:
            try:
                item = resultCOM_queue.get(timeout=10)
                if item is None:
                    sentinel_count += 1
                else:
                    output.append(item)
            except queue.Empty:
                print(f"[SaveToTxt] Timeout waiting for COM results")
                break

        print(f"[SaveToTxt] Collected {len(output)} COM results")

        # Wait for workers
        for p in workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join()

        writer.join(timeout=10)
        if writer.is_alive():
            writer.terminate()

        # Save COM data
        if output:
            output.sort(key=lambda x: x["frame_index"])
            df = pd.DataFrame(output)
            df.to_csv(filename, index=False)
            print(f"[SaveToTxt] Saved COM data to {filename}")

        endTime = time.time()
        print(f"[SaveToTxt] Total time: {endTime - startTime:.1f}s")
    
    ##### Updated version above #### Old version below (commented out) #####
    # def SaveToTxt(self, sex, filename, confidencelevel=0.85, displayCOM=False):
    #     startTime = time.time()

    #     cfg = dict(
    #         sex=sex, confidence=confidencelevel, displayCOM=displayCOM,
    #         fps=self.fps, frame_count=self.frame_count, frame_width=self.frame_width, frame_height=self.frame_height, 
    #         lag=self.lag, corners=self.corners,
    #         fx1=self.fx1, fx2=self.fx2, fy1=self.fy1, fy2=self.fy2,
    #         fz1=self.fz1, fz2=self.fz2, px1=self.px1, px2=self.px2,
    #         py1=self.py1, py2=self.py2
    #     )

    #     frame_queue = Queue()  
    #     resultCOM_queue = Queue()
    #     resultVector_queue = Queue()

    #     num_workers = min(6, processing.cpu_count())

    #     reader_thread = threading.Thread(target=frame_reader, args=(frame_queue, self.video_path, self.reader_start_frame, num_workers, self.effective_frames, cfg))
    #     reader_thread.start()

    #     writer = processing.Process(
    #         target=writer_process,
    #         args=(resultVector_queue, self.output_mp4, self.fps, (self.frame_width, self.frame_height), num_workers),
    #         daemon=True
    #     )
    #     writer.start()


    #     # Start worker processes
    #     workers: list[Process] = []
    #     for i in range(num_workers):
    #         print(f"[PROCESS] Starting process {i}")
    #         p = processing.Process(target=process_frame,
    #                    args=(frame_queue, resultVector_queue, resultCOM_queue, cfg),
    #                    daemon=True)
    #         p.start()
    #         workers.append(p)

    #     reader_thread.join()
    #     print("[MAIN] Reader thread finished")

    #     print(f"Size of result_queue: {resultCOM_queue.qsize()}")
    #     # Drain result queue
    #     output = []
    #     sentinel_count = 0
    #     while sentinel_count < num_workers:
    #         try:
    #             item = resultCOM_queue.get() # Add a timeout here for debugging
    #             if item is None:
    #                 sentinel_count += 1
    #                 print(f"[MAIN] Received sentinel from worker ({sentinel_count}/{num_workers})")
    #             else:
    #                 output.append(item)
    #         except Exception as e:
    #             print(f"[MAIN ERROR] Timeout or error while draining result queue: {e}")
    #             print(f"[MAIN ERROR] Current sentinel count: {sentinel_count}/{num_workers}")
    #             print(f"[MAIN ERROR] Remaining workers not yet joined: {[p.name for p in workers if p.is_alive()]}")
    #             break # Break the loop if we timeout
    #     print(f"[MAIN] Drained all results from queue. Total items: {len(output)}")

    #     print("[MAIN] Waiting for all worker processes to finish joining...")
    #     joined_workers_count = 0
    #     for p in workers:
    #         print(f"[MAIN] Attempting to join worker {p.name} (PID: {p.pid})...")
    #         p.join() # Try to join with a timeout

    #         if p.is_alive():
    #             print(f"[MAIN ERROR] Worker {p.name} (PID: {p.pid}) did NOT join after timeout (still alive). Forcibly terminating...")
    #             p.terminate() # <-- Forcefully terminate the rogue process
    #             p.join() # Give it a moment to terminate after being signaled
    #             if p.is_alive():
    #                 print(f"[MAIN ERROR] Worker {p.name} (PID: {p.pid}) *still* alive after forceful termination!")
    #             else:
    #                 print(f"[MAIN] Worker {p.name} (PID: {p.pid}) successfully terminated after force.")
    #                 joined_workers_count += 1 # Count it as "handled" for joining purposes
    #         else:
    #             # Process successfully joined within the timeout
    #             joined_workers_count += 1
    #             print(f"[MAIN] Worker {p.name} (PID: {p.pid}) finished joining.")

                
    #     writer.join()

    #     output.sort(key=lambda x: x["frame_index"])
    #     df = pd.DataFrame(output)
    #     df.to_csv(filename, index=False)

    #     endTime = time.time()
    #     print("[INFO] COM Total time taken: ", endTime - startTime)

    #     count = 0
    #     vector_cam = cv2.VideoCapture(self.output_mp4)
    #     com_helper = COM_helper()
    #     while(vector_cam.isOpened() and count < self.effective_frames):
    #         ret, frame = vector_cam.read()
    #         frame = com_helper.drawFigure(frame, count)
    #         cv2.imshow("Long View", cv2.resize(frame, (int(self.frame_width * 0.5), int(self.frame_height * 0.5))))
    #         if cv2.waitKey(1) & 0xFF == ord("q"):
    #                 break
    #         count += 1

    #     vector_cam.release()
    #     cv2.destroyAllWindows()
        



# if __name__ == "__main__":
#     video_path = r"C:\Users\Student\Downloads\spu_lr_NS_long_vid01.mov"  # Change this to your video file path
#     #cap = cv2.VideoCapture(video_path)
#     print(f"DEBUG IN MAIN: Type of video_path before Processor: {type(video_path)}") # Add this
#     print(f"DEBUG IN MAIN: Value of video_path before Processor: '{video_path}'") # Add this

#     processor = Processor(video_path)

#     # Example usage:
#     # Draw skeleton with display options:
#     # processor.find_coordinates(sex='male', video_path, filename='output.mp4', displayname=True, displaystickfigure=True, displayCOM=True)

#     # Save landmarks to CSV with zeros for no detection
#     processor.SaveToTxt(sex='male', filename='pose_landmarks.csv', confidencelevel=0.85, displayCOM=True)