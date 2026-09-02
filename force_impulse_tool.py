import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

sys.path.append(str(Path(__file__).resolve().parent))

from Util.fileFormater import FileFormatter
from Util.force_boundary_finder import find_force_boundaries

_trapz = getattr(np, "trapezoid", None) or np.trapz

FORCE_THRESHOLD = 50
TIME_COL = "Time(s)"
MAX_DRAW_POINTS = 20000

force_dict = {'abs time (s)': 'Time(s)',
              'Fx': 'FP1_Fx', 'Fy': 'FP1_Fy', 'Fz': 'FP1_Fz', '|Ft|': 'FP1_|F|', 'Ax': 'FP1_Ax', 'Ay': 'FP1_Ay',
              'Fx.1': 'FP2_Fx', 'Fy.1': 'FP2_Fy', 'Fz.1': 'FP2_Fz', '|Ft|.1': 'FP2_|F|', 'Ax.1': 'FP2_Ax', 'Ay.1': 'FP2_Ay',
              'Fx.2': 'FP3_Fx', 'Fy.2': 'FP3_Fy', 'Fz.2': 'FP3_Fz', '|Ft|.2': 'FP3_|F|', 'Ax.2': 'FP3_Ax', 'Ay.2': 'FP3_Ay'}

formatter_dict = {'abs time (s)': 'Time(s)',
                  'Fx1': 'FP1_Fx', 'Fy1': 'FP1_Fy', 'Fz1': 'FP1_Fz', '|Ft1|': 'FP1_|F|', 'Ax1': 'FP1_Ax', 'Ay1': 'FP1_Ay',
                  'Fx2': 'FP2_Fx', 'Fy2': 'FP2_Fy', 'Fz2': 'FP2_Fz', '|Ft2|': 'FP2_|F|', 'Ax2': 'FP2_Ax', 'Ay2': 'FP2_Ay'}


def read_named_columns(path_force):
    df_force = pd.read_csv(path_force, header=17, delimiter='\t', encoding='latin1').drop(0)
    df_force = df_force.apply(pd.to_numeric, errors='coerce')
    df_force.rename(columns=force_dict, inplace=True)
    return df_force, "named columns"


def read_with_formatter(path_force):
    reader = FileFormatter()
    if path_force.endswith('.txt'):
        df_force = reader.readTxt(path_force)
    elif path_force.endswith('.csv'):
        df_force = reader.readCsv(path_force)
    else:
        df_force = reader.readExcel(path_force)
    df_force = df_force.apply(pd.to_numeric, errors='coerce')
    df_force.rename(columns=formatter_dict, inplace=True)
    return df_force, "FileFormatter"


def force_channels(df_force):
    names = []
    for plate in ("FP1", "FP2", "FP3"):
        for component in ("Fx", "Fy", "Fz", "|F|"):
            col = plate + "_" + component
            if col in df_force.columns and df_force[col].notna().any():
                names.append(col)
    return names


def load_force_file(path_force):
    if path_force.endswith('.txt'):
        readers = [read_named_columns, read_with_formatter]
    else:
        readers = [read_with_formatter]

    problems = []
    for reader in readers:
        try:
            df_force, how = reader(path_force)
            if TIME_COL not in df_force.columns:
                raise ValueError("no time column")
            df_force = df_force.dropna(subset=[TIME_COL]).reset_index(drop=True)
            steps = np.diff(df_force[TIME_COL].to_numpy(dtype=float))
            if len(df_force) < 2 or not np.all(steps >= 0):
                raise ValueError("time column does not increase")
            if not force_channels(df_force):
                raise ValueError("no force columns found")
            print(f"[INFO] Read {len(df_force)} rows using {how}")
            return df_force, how
        except Exception as e:
            print(f"[WARN] {reader.__name__} failed: {e}")
            problems.append(reader.__name__ + ": " + str(e))

    raise RuntimeError("Could not read this force file.\n" + "\n".join(problems))


def area_under_curve(t, y, marker_a, marker_b):
    low, high = min(marker_a, marker_b), max(marker_a, marker_b)
    inside = (t >= low) & (t <= high) & np.isfinite(y)
    tt, yy = t[inside], y[inside]
    if tt.size < 2:
        return 0.0, 0.0, 0, float("nan")
    area = float(_trapz(yy, tt))
    peak = float(yy[np.argmax(np.abs(yy))])
    return area, float(tt[-1] - tt[0]), int(tt.size), peak


class ImpulseTool(tk.Tk):
    def __init__(self, path_force=None):
        super().__init__()
        self.title("Force Impulse Tool")
        self.geometry("1050x740")

        self.Force = None
        self.t = np.array([])
        self.y = np.array([])
        self.dragging = None

        self.build_ui()

        if path_force:
            self.after(50, lambda: self.upload_force_data(path_force))

    def build_ui(self):
        bar = tk.Frame(self, padx=10, pady=8)
        bar.pack(fill=tk.X)

        tk.Button(bar, text="Upload force data", command=self.uploadForceCallback).pack(side=tk.LEFT)
        tk.Label(bar, text="  Channel:").pack(side=tk.LEFT)
        self.channel = tk.StringVar()
        self.channel_menu = tk.OptionMenu(bar, self.channel, "")
        self.channel_menu.config(width=10)
        self.channel_menu.pack(side=tk.LEFT)
        tk.Button(bar, text="Snap to contact", command=self.snap_to_contact).pack(side=tk.LEFT, padx=8)
        tk.Button(bar, text="Reset", command=self.reset_markers).pack(side=tk.LEFT)

        self.file_label = tk.Label(bar, text="No force data loaded", anchor="e", fg="#555")
        self.file_label.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(10, 4.6))
        self.fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.13)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8)

        toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X, padx=8)

        self.curve, = self.ax.plot([], [], linewidth=1.0, color='purple')
        self.shaded = self.ax.axvspan(0, 0, color='purple', alpha=0.15)
        self.marker_a = self.ax.axvline(0, color='blue', linewidth=1.6)
        self.marker_b = self.ax.axvline(0, color='red', linewidth=1.6)
        self.ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Force (N)")
        self.ax.grid(True, alpha=0.3)

        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        self.result = tk.StringVar(value="Upload a force file to begin.")
        tk.Label(self, textvariable=self.result, justify=tk.LEFT, anchor="w",
                 font=("Menlo", 13), padx=10, pady=10).pack(fill=tk.X)

    def uploadForceCallback(self):
        file_path = filedialog.askopenfilename(
            title="Select Force Data File",
            filetypes=[("Force data", "*.txt *.csv *.xlsx *.xls"), ("All files", "*.*")])
        if file_path:
            self.upload_force_data(file_path)

    def upload_force_data(self, file_path):
        print(f"[INFO] Force data uploaded: {file_path}")
        try:
            df_force, how = load_force_file(file_path)
        except Exception as e:
            print(f"[ERROR] Failed to read force data: {e}")
            messagebox.showerror("Could not load file", str(e))
            return

        self.Force = df_force
        names = force_channels(df_force)

        menu = self.channel_menu["menu"]
        menu.delete(0, "end")
        for name in names:
            menu.add_command(label=name, command=lambda v=name: self.plot_force_data(v))

        steps = np.diff(df_force[TIME_COL].to_numpy(dtype=float))
        hz = 1.0 / np.median(steps[steps > 0])
        print(f"[DEBUG] num of rows: {len(df_force)}")
        print(f"[DEBUG] Approximate sampling rate: {hz:.2f} Hz")
        self.file_label.config(
            text=f"{os.path.basename(file_path)}  |  {len(df_force):,} rows  |  ~{hz:,.0f} Hz  |  {how}")

        self.plot_force_data("FP1_Fz" if "FP1_Fz" in names else names[0])

    def plot_force_data(self, name):
        self.channel.set(name)
        self.t = self.Force[TIME_COL].to_numpy(dtype=float)
        self.y = self.Force[name].to_numpy(dtype=float)

        step = max(1, len(self.t) // MAX_DRAW_POINTS)
        self.curve.set_data(self.t[::step], self.y[::step])

        low, high = np.nanmin(self.y), np.nanmax(self.y)
        pad = 0.08 * (high - low) if high > low else 1.0
        self.ax.set_xlim(self.t[0], self.t[-1])
        self.ax.set_ylim(low - pad, high + pad)
        self.ax.set_title(name, fontsize=10)
        self.reset_markers()

    def marker_x(self, which):
        line = self.marker_a if which == "a" else self.marker_b
        return float(np.atleast_1d(line.get_xdata())[0])

    def set_marker(self, which, x):
        x = float(np.clip(x, self.t[0], self.t[-1]))
        line = self.marker_a if which == "a" else self.marker_b
        line.set_xdata([x])

    def reset_markers(self):
        if self.t.size < 2:
            return
        first, last = self.t[0], self.t[-1]
        self.set_marker("a", first + 0.25 * (last - first))
        self.set_marker("b", first + 0.75 * (last - first))
        self.refresh()

    def snap_to_contact(self):
        if self.Force is None:
            return
        df_force = self.Force.copy()
        df_force["FrameNumber"] = df_force.index
        try:
            start, end = find_force_boundaries(df_force, threshold=FORCE_THRESHOLD, padding_frames=0)
        except Exception as e:
            print(f"[WARN] Could not find force boundaries: {e}")
            return
        self.set_marker("a", self.t[int(start)])
        self.set_marker("b", self.t[int(end)])
        self.refresh()

    def on_press(self, event):
        if event.inaxes is not self.ax or event.xdata is None or self.t.size < 2:
            return
        if self.canvas.toolbar.mode:
            return
        pixel_a = self.ax.transData.transform((self.marker_x("a"), 0))[0]
        pixel_b = self.ax.transData.transform((self.marker_x("b"), 0))[0]
        self.dragging = "a" if abs(event.x - pixel_a) <= abs(event.x - pixel_b) else "b"
        self.set_marker(self.dragging, event.xdata)
        self.refresh()

    def on_motion(self, event):
        if self.dragging and event.inaxes is self.ax and event.xdata is not None:
            self.set_marker(self.dragging, event.xdata)
            self.refresh()

    def on_release(self, event):
        self.dragging = None

    def refresh(self):
        a, b = self.marker_x("a"), self.marker_x("b")
        low, high = min(a, b), max(a, b)
        self.shaded.set_xy([[low, 0], [low, 1], [high, 1], [high, 0], [low, 0]])

        area, duration, n, peak = area_under_curve(self.t, self.y, a, b)
        mean = area / duration if duration > 0 else float("nan")
        self.result.set(
            f"Area = {area:,.2f} N*s      {low:.4f} to {high:.4f} s      "
            f"{duration * 1000:,.1f} ms, {n:,} samples\n"
            f"Mean {mean:,.2f} N      Peak {peak:,.2f} N      Channel {self.channel.get()}")
        self.canvas.draw_idle()


def main():
    path_force = sys.argv[1] if len(sys.argv) > 1 else None
    if path_force and not os.path.exists(path_force):
        print(f"[ERROR] File not found: {path_force}")
        return 1
    ImpulseTool(path_force).mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
