import tkinter as tk
import threading
from vector_overlay.stick_figure_COM import Processor

def COMCallback(self):
    print("[INFO] COM processing...")
    processor = Processor(self.Video.cam)
    def threadTarget():
        print("[INFO] Starting COM processing in a separate thread...")
        processor.SaveToTxt(sex="m", filename="pose_landmarks.csv", confidencelevel=0.85, displayCOM=True)
        print("[INFO] COM processing finished.")
        self.COM_flag = True
    
    COMThread = threading.Thread(target=threadTarget, daemon=True)
    COMThread.start()


    
