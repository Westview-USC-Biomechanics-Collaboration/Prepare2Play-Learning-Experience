import tkinter as tk
import threading
import cv2
from vector_overlay.stick_figure_COM import Processor

def COMCallback(self, s):
    print("[INFO] COM processing...")
    
    def threadTarget():
        print("[INFO] Starting COM processing in a separate thread...")
        processor = Processor(self.Video.path)
        processor.SaveToTxt(sex=s, filename="pose_landmarks.csv", confidencelevel=0.85, displayCOM=True)
        print("[INFO] COM processing finished.")
        self.COM_flag = True

    
    COMThread = threading.Thread(target=threadTarget, daemon=True)
    COMThread.start()


    