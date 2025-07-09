import tkinter as tk
import threading
import cv2
from vector_overlay.stick_figure_COM import Processor

def COMCallback(self):
    print("[INFO] COM processing...")
    
    def threadTarget():
        print("[INFO] Starting COM processing in a separate thread...")
        copyCam = cv2.VideoCapture(self.Video.path)
        copyCam.set(cv2.CAP_PROP_POS_FRAMES, 0)
        processor = Processor(self.Video.path)
        processor.SaveToTxt(sex="f", filename="pose_landmarks.csv", confidencelevel=0.85, displayCOM=True)
        copyCam.release()
        del copyCam
        print("[INFO] COM processing finished.")
        self.COM_flag = True

    
    COMThread = threading.Thread(target=threadTarget, daemon=True)
    COMThread.start()


    
