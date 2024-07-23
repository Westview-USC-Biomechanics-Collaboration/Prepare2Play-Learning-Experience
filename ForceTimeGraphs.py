import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import PIL.Image, PIL.ImageTk
import threading
from scipy.signal import find_peaks as fp
from scipy.signal import savgol_filter
from peakutils import baseline
import mplcursors

def select_excel_file():
    global file_path, df1, x, y, y2, y3, y4, fig, ax, fig2, ax2, line1, line2, line3, line4, cursorLine, cursorLine2, canvas, canvas2
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        df1 = pd.read_excel(file_path, sheet_name='ajp_lr_JN_for01', skiprows=list(range(17)) + [18])

        # Axis for first graph
        x = df1['abs time (s)']
        y = df1['Fz']
        y2 = df1['Fy']

        # Axis for second graph
        y3 = df1.iloc[:, 12]
        y4 = df1.iloc[:, 11]

        # Update plots
        fig, ax = plt.subplots(figsize=(10, 5))
        line1, = ax.plot(x, y, color='Skyblue', label='Fz')
        line2, = ax.plot(x, y2, color='red', label='Fy')
        cursorLine = ax.axvline(x=0, color='k', linestyle='--')

        # Find peaks for left
        peaks_y, _ = fp(y, height=1000, distance=20)
        peaks_y2, _ = fp(y2, height=50, distance=20)

        # Plot peaks for left
        ax.plot(x[peaks_y], y[peaks_y], "x", label='Fz Peaks')
        ax.plot(x[peaks_y2], y2[peaks_y2], "o", label='Fy Peaks')

        # Axis for left
        ax.set_xlabel('abs time (s)')
        ax.set_ylabel('Fz + Fy')
        ax.set_title('Title')
        ax.legend()

        # Cursor for left
        cursor1 = mplcursors.cursor(line1, hover=True)
        cursor1.connect("add", lambda sel: sel.annotation.set_text(f'x: {sel.target[0]:.3f}\ny: {sel.target[1]:.3f}'))

        cursor2 = mplcursors.cursor(line2, hover=True)
        cursor2.connect("add", lambda sel: sel.annotation.set_text(f'x: {sel.target[0]:.3f}\ny: {sel.target[1]:.3f}'))

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.LEFT)

        # Add Lines for right
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        line3, = ax2.plot(x, y3, color='Skyblue', label='Fz')
        line4, = ax2.plot(x, y4, color='red', label='Fy')
        cursorLine2 = ax2.axvline(x=0, color='k', linestyle='--')

        # Find peaks for right
        peaks_y3, _ = fp(y3, height=1000, distance=20)
        peaks_y4, _ = fp(y4, height=50, distance=20)

        # Plot peaks for right
        ax2.plot(x[peaks_y3], y3[peaks_y3], "x", label='Fz Peaks')
        ax2.plot(x[peaks_y4], y4[peaks_y4], "o", label='Fy Peaks')

        # Axis for right
        ax2.set_xlabel('abs time (s)')
        ax2.set_ylabel('Fz + Fy')
        ax2.set_title('Title')
        ax2.legend()

        # Cursor for right
        cursor3 = mplcursors.cursor(line3, hover=True)
        cursor3.connect("add", lambda sel: sel.annotation.set_text(f'x: {sel.target[0]:.3f}\ny: {sel.target[1]:.3f}'))

        cursor4 = mplcursors.cursor(line4, hover=True)
        cursor4.connect("add", lambda sel: sel.annotation.set_text(f'x: {sel.target[0]:.3f}\ny: {sel.target[1]:.3f}'))

        canvas2 = FigureCanvasTkAgg(fig2, master=root)
        canvas_widget2 = canvas2.get_tk_widget()
        canvas_widget2.pack(side=tk.RIGHT)

def select_video_file():
    global video_path, cap, video_duration
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if video_path:
        cap = cv2.VideoCapture(video_path)
        video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        display_video()

def display_video():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        img = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        videoLabel.imgtk = img
        videoLabel.configure(image=img)
        root.after(10, display_video)

def updateFrame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))

        curTime = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        progress = curTime / video_duration * 100
        bar['value'] = progress

        # Progress in data
        dataDuration = x.max()
        dataProgress = (curTime / video_duration) * dataDuration
        cursorLine.set_xdata([dataProgress, dataProgress])
        cursorLine2.set_xdata([dataProgress, dataProgress])

        canvas.draw()
        canvas2.draw()

        # Find the data
        closest_idx = (x - dataProgress).abs().idxmin()
        fz_value = y.iloc[closest_idx]

    root.after(10, updateFrame)

def start_video():
    if cap is not None and cap.isOpened():
        threading.Thread(target=updateFrame).start()

root = tk.Tk()
root.title("Video and Graph Sync")

# File selection buttons
file_button = tk.Button(root, text="Select Excel File", command=select_excel_file)
file_button.pack()

video_button = tk.Button(root, text="Select Video File", command=select_video_file)
video_button.pack()

start_button = tk.Button(root, text="Start Video", command=start_video)
start_button.pack()

# Video display
videoFrame = tk.Frame(root)
videoFrame.pack()

videoLabel = ttk.Label(videoFrame)
videoLabel.pack()

bar = ttk.Progressbar(videoFrame, orient='horizontal', length=500, mode='determinate')
bar.pack()

root.mainloop()

if cap:
    cap.release()
