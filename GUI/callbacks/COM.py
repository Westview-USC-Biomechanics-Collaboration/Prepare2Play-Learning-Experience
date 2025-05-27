import tkinter as tk
import threading
from vector_overlay.stick_figure_COM import Processor

def COMCallback(self):
    print("[INFO] COM processing...")
    processor = Processor(self.Video.cam)
    def threadTarget():
        processor.SaveToTxt(sex="m", filename="test0721.mp4", confidencelevel=0.85)
        self.COM_flag = True
    
    COMThread = threading.Thread(target=threadTarget, daemon=True)
    COMThread.start()


    
