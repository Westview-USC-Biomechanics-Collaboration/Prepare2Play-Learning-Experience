#1 Starts by taking in .txt file and video fil and calling the ledSyncing.py script
import pandas as pd
import os 
import sys
import tkinter as tk

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ledSyncing import run_led_syncing

data_path = r"C:\Users\berke\OneDrive\Desktop\USCBiomechanicsProject\Prepare2Play-Learning-Experience\newData"
video_file = "walk_test_vid01.mov"
force_file = "walktest1.txt"

run_led_syncing(data_path, video_file, force_file)


#2 Gather lag from video_file_Results.csv and print it

def get_lag_from_results(parent_path, force_file):
    df_result = pd.read_csv(os.path.join(parent_path, force_file.replace('.txt', '_Results.csv')))
    lag_value = df_result['Video Frame for t_zero force'].values[0]
    return int(lag_value)

#3 Pass lag value into simplerGUI.py

from simplerGUI import VectorOverlayApp

lag_value = get_lag_from_results(data_path, force_file)

root = tk.Tk()
app = VectorOverlayApp(root)
app.lagValue = lag_value  # Pass the lag value to the GUI

# Handle window closing
def on_closing():
    import cv2
    cv2.destroyAllWindows()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()