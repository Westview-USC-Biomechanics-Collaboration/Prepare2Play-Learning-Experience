"""
Modified COM Processor that respects boundary frames.
This wraps the existing Processor class to add boundary support.
"""
import cv2
import time
import pandas as pd
import threading
import multiprocessing as processing
from multiprocessing import Process, Queue
import queue

# Import the correct process_frame function from stick_figure_COM
from vector_overlay.stick_figure_COM import (
    process_frame  # This expects: (q, results_queue, sex, confidencelevel, displayCOM)
)


class BoundaryProcessor:
    """
    Wrapper around stick_figure_COM.Processor that adds boundary frame support.
    """
    
    def __init__(self, video_path):
        self.video_path = video_path
        
        # Open video to get metadata
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    
    def SaveToTxt(self, sex, filename, confidencelevel=0.85, displayCOM=False,
                  start_frame=None, end_frame=None, max_workers=3):   # Change start frame to 0 (default)
        """
        Save pose landmarks and COM to CSV, processing only frames in boundary.
        
        Args:
            sex: 'm' or 'f' for male/female
            filename: Output CSV filename
            confidencelevel: Minimum confidence for pose detection
            displayCOM: Whether to calculate and save COM coordinates
            start_frame: First frame to process (inclusive)
            end_frame: Last frame to process (inclusive), None means process to end
            max_workers: Maximum number of worker processes
        
        Returns:
            str: Path to saved CSV file
        """
        startTime = time.time()
        
        if end_frame is None:
            end_frame = self.frame_count - 1
        
        # Validate boundaries
        start_frame = max(0, start_frame)
        end_frame = min(self.frame_count - 1, end_frame)
        
        print(f"\n========== COM Calculation ==========")
        print(f"Video: {self.video_path}")
        print(f"Processing frames: {start_frame} to {end_frame}")
        print(f"Total frames to process: {end_frame - start_frame + 1}")
        print(f"Workers: {max_workers}")
        print(f"Sex: {sex}, Confidence: {confidencelevel}, Display COM: {displayCOM}")
        print("=" * 50)
        
        # Configuration for workers
        cfg = dict(
            sex=sex,
            confidence=confidencelevel,
            displayCOM=displayCOM,
            fps=self.fps,
            frame_count=self.frame_count,
            frame_width=self.frame_width,
            frame_height=self.frame_height
        )
        
        # Create queues
        frame_queue = Queue()
        result_queue = Queue()
        
        # Start frame reader thread
        reader_thread = threading.Thread(
            target=self._boundary_frame_reader,
            args=(frame_queue, start_frame, end_frame, max_workers)
        )
        reader_thread.start()
        
        # Start worker processes
        workers = []
        for i in range(max_workers):
            print(f"[MAIN] Starting COM worker {i}")
            # Pass individual arguments as expected by process_frame
            p = processing.Process(
                target=process_frame,
                args=(frame_queue, result_queue, sex, confidencelevel, displayCOM),
                daemon=True
            )
            p.start()
            workers.append(p)
        
        # Wait for reader to finish
        reader_thread.join()
        print("[MAIN] Frame reader finished")
        
        # Collect results
        output = []
        sentinel_count = 0
        expected_sentinels = max_workers
        
        while sentinel_count < expected_sentinels:
            try:
                item = result_queue.get(timeout=30)
                if item is None:
                    sentinel_count += 1
                    print(f"[MAIN] Received sentinel ({sentinel_count}/{expected_sentinels})")
                else:
                    output.append(item)
            except queue.Empty:
                print(f"[MAIN] Timeout waiting for results (got {len(output)} so far)")
                break
        
        print(f"[MAIN] Collected {len(output)} results from queue")
        
        # Wait for workers to finish
        for i, p in enumerate(workers):
            print(f"[MAIN] Joining worker {i}...")
            p.join(timeout=10)
            if p.is_alive():
                print(f"[MAIN] Worker {i} still alive, terminating...")
                p.terminate()
                p.join()
        
        # Sort by frame index and save
        output.sort(key=lambda x: x["frame_index"])
        df = pd.DataFrame(output)
        df.to_csv(filename, index=False)
        
        endTime = time.time()
        print(f"[INFO] COM processing complete in {endTime - startTime:.2f}s")
        print(f"[INFO-COM_Processor_Modified] Saved {len(df)} frames to {filename}")
        print("=" * 50 + "\n")
        
        return filename
    
    def _boundary_frame_reader(self, frame_queue, start_frame, end_frame, num_workers):
        """
        Read frames within the specified boundary and put them in the queue.
        """
        print(f"[READER] Opening video: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"[READER ERROR] Could not open video")
            for _ in range(num_workers):
                frame_queue.put(None)
            return
        
        # Jump to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        
        print(f"[READER] Reading frames {start_frame} to {end_frame}...")
        frames_read = 0
        
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                print(f"[READER] Could not read frame {current_frame}")
                break
            
            try:
                # Use relative index for output (0-based from start_frame)
                frame_queue.put((current_frame, frame.copy()), timeout=5)
                frames_read += 1
                
                if frames_read % 100 == 0:
                    print(f"[READER] Queued {frames_read} frames...")
                
            except queue.Full:
                print(f"[READER] Queue full, waiting...")
                time.sleep(0.1)
                continue
            
            current_frame += 1
        
        cap.release()
        print(f"[READER] Finished reading {frames_read} frames. Sending sentinels...")
        
        # Send sentinels
        for _ in range(num_workers):
            frame_queue.put(None)
        
        print("[READER] Done")