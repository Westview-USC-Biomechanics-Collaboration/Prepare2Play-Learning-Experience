import tkinter as tk
import threading
import cv2
from vector_overlay.stick_figure_COM import Processor
from vector_overlay.vectoroverlay_GUI import VectorOverlay
from GUI.callbacks.global_variable import globalVariable

def COMCallback(self, s):
    """
    Process COM using the trimmed video if available.
    
    Args:
        s: Sex ('m' or 'f') for body segment calculations
    """
    print(f"[INFO] Starting COM processing with sex={s}...")
    
    def threadTarget():
        print("[INFO] COM processing in separate thread...")
        
        # Check if we have a trimmed video from vector overlay
        if hasattr(self.Video, 'trimmed_path') and self.Video.trimmed_path:
            video_path = self.Video.trimmed_path
            print(f"[INFO] Using trimmed video: {video_path}")
        else:
            video_path = self.Video.path
            print(f"[INFO] Using original video: {video_path}")
        
        # Get lag from alignment
        try:
            with open("lag.txt", "r") as f:
                lag = int(f.read().strip())
            print(f"[INFO] Using lag from alignment: {lag}")
        except FileNotFoundError:
            lag = 0
            print("[WARNING] No lag.txt found, using lag=0")
        
        # Check if we have aligned force data
        if hasattr(self.state, 'df_aligned') and self.state.df_aligned is not None:
            force_data = self.state.df_aligned
            print(f"[INFO] Using aligned force data: {len(force_data)} rows")
        else:
            force_data = self.Force.data
            print(f"[INFO] Using raw force data: {len(force_data)} rows")
        
        # Create processor
        processor = Processor(
            video_path=video_path
            # data_df=force_data,
            # lag=lag,
            # output_mp4="vector_overlay_temp.mp4"
        )
        
        # Run COM processing
        processor.SaveToTxt(
            sex=s, 
            filename='pose_landmarks.csv', 
            confidencelevel=0.85, 
            displayCOM=True
        )
        
        print("[INFO] COM processing finished.")
        self.state.com_enabled = True

    COMThread = threading.Thread(target=threadTarget, daemon=True)
    COMThread.start()

# def COMCallback(self, s):
#     print("[INFO] COM processing...")
    
#     def threadTarget():
#         print("[INFO] Starting COM processing in a separate thread...")
#         processor = Processor(self.Video.path)
#         processor.SaveToTxt(sex=s, filename="pose_landmarks.csv", confidencelevel=0.85, displayCOM=True)
#         print("[INFO] COM processing finished.")
#         self.COM_flag = True
#         # globalVariable.sex = s

#     COMThread = threading.Thread(target=threadTarget, daemon=True)
#     COMThread.start()


    