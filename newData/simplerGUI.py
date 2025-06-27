import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import pandas as pd
import sys
sys.path.append(r"C:\Users\gulbd\OneDrive\Documents\GitHub\Prepare2Play-Learning-Experience")

from vector_overlay.vectoroverlay_GUI import VectorOverlay

# ---- Main GUI App ----
class VectorOverlayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vector Overlay Generator")
        self.root.geometry("500x300")

        self.video_path = None
        self.csv_path = None
        self.view_option = tk.StringVar(value="Long")

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Select Video File").pack(pady=5)
        tk.Button(self.root, text="Browse Video", command=self.browse_video).pack()

        tk.Label(self.root, text="Select Force Data CSV").pack(pady=5)
        tk.Button(self.root, text="Browse CSV", command=self.browse_csv).pack()

        tk.Label(self.root, text="Select View Mode").pack(pady=5)
        ttk.Combobox(self.root, textvariable=self.view_option, values=["Long", "Top", "Short"]).pack()

        tk.Button(self.root, text="Run Overlay Visualization", command=self.run_overlay).pack(pady=20)

    def browse_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.MOV")])
        if file_path:
            self.video_path = file_path
            # Removed messagebox popup

    def browse_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.csv_path = file_path
            # Removed messagebox popup

    def run_overlay(self):
        if not self.video_path or not self.csv_path:
            messagebox.showerror("Missing File", "Please select both video and CSV files.")
            return

        try:
            df = pd.read_csv(self.csv_path)
            cap = cv2.VideoCapture(self.video_path)

            # No manual corner selection here, VectorOverlay handles it internally
            overlay = VectorOverlay(df, cap)

            view_mode = self.view_option.get().lower()
            if view_mode == "long":
                overlay.LongVectorOverlay(outputName=None)  # Live preview only
            elif view_mode == "top":
                overlay.TopVectorOverlay(outputName=None)
            elif view_mode == "short":
                overlay.ShortVectorOverlay(outputName=None)
            else:
                messagebox.showerror("Invalid View Mode", "Select a valid view mode.")
                return

        except Exception as e:
            messagebox.showerror("Error", f"An error!!!!!!:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VectorOverlayApp(root)
    root.mainloop()
