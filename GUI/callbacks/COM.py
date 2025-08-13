import tkinter as tk
import threading
import cv2
from vector_overlay.stick_figure_COM import Processor
from vector_overlay.vectoroverlay_GUI import VectorOverlay

def COMCallback(self):
    print("[INFO] COM processing...")
    
    def threadTarget():
        print("[INFO] Starting COM processing in a separate thread...")
        with open("lag.txt", "r") as f:
            lag = int(f.read().strip())
            print(lag)

        copyCam = cv2.VideoCapture(self.Video.path)
        copyCam.set(cv2.CAP_PROP_POS_FRAMES, lag)
        processor = Processor(copyCam)
        processor.SaveToTxt(sex="m", filename="pose_landmarks.csv", confidencelevel=0.85, displayCOM=True)
        copyCam.release()
        del copyCam
        print("[INFO] COM processing finished.")
        self.state.com_enabled = True

    
    COMThread = threading.Thread(target=threadTarget, daemon=True)
    COMThread.start()


    
