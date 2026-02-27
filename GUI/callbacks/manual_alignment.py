import os
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class AlignmentGUI:
    def __init__(self, root, video=None, force=None):
        """
        Parameters
        ----------
        video  : VideoState object (has .path, .cam, .fps, .total_frames)
                 or a plain path string
        force  : Force object (has .data as a pandas DataFrame)
                 or a plain path string
        """
        self.root = root
        self.root.title("Video & Force Data Alignment Tool")
        self.root.geometry("1200x750")

        self.cap = None
        self._owns_cap = False          # True only if WE opened the capture (so we close it)
        self.force_time = None
        self.force_data = None
        self.offset = 0.0
        self.video_fps = 30
        self.current_frame = 0
        self.playing = False

        self._build_ui()

        # ── Auto-load video ───────────────────────────────────────────────
        if video is not None:
            self._init_video(video)
            print(f"Loaded video: {video.path if hasattr(video, 'path') else str(video)}")

        # ── Auto-load force ───────────────────────────────────────────────
        if force is not None:
            self._init_force(force)
            print(f"Loaded force data: {force.path if hasattr(force, 'path') else str(force)}")

    # ── Init from objects ─────────────────────────────────────────────────
    def _init_video(self, video):
        """Accept a VideoState object or a plain path string."""
        if hasattr(video, "cam") and video.cam is not None and video.cam.isOpened():
            # Reuse the already-open VideoCapture from VideoState
            self.cap = video.cam
            self._owns_cap = False
            self.video_fps = video.fps if hasattr(video, "fps") and video.fps else 30
            total = video.total_frames if hasattr(video, "total_frames") else int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            # Fall back to opening by path
            path = video.path if hasattr(video, "path") else str(video)
            if not path or not os.path.exists(path):
                messagebox.showerror("File Not Found", f"Video file not found:\n{path}")
                return
            self.cap = cv2.VideoCapture(path)
            self._owns_cap = True
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.frame_slider.config(to=max(total - 1, 1))
        self.frame_var.set(0)

        name = ""
        if hasattr(video, "path") and video.path:
            name = os.path.basename(video.path)
            self.root.title(f"Alignment Tool — {name}")

        self.show_frame(0)

    def _init_force(self, force):
        """Accept a Force object (with .data DataFrame) or a plain path string."""
        if hasattr(force, "data") and force.data is not None:
            self._load_force_from_dataframe(force.data)
        elif hasattr(force, "path") and force.path:
            self._load_force_from_path(force.path)
        else:
            self._load_force_from_path(str(force))

    # ── UI Layout ─────────────────────────────────────────────────────────
    def _build_ui(self):
        # Top button bar
        btn_frame = tk.Frame(self.root, pady=6, padx=6)
        btn_frame.pack(side="top", fill="x")

        tk.Button(btn_frame, text="Load Video",       command=self.load_video,       width=14).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Load Force Data",  command=self.load_force,       width=14).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Export Alignment", command=self.export_alignment, width=14).pack(side="right", padx=4)

        # Bottom controls — packed BEFORE content so they're never clipped
        ctrl = tk.Frame(self.root, pady=6, padx=6, relief="groove", bd=1)
        ctrl.pack(side="bottom", fill="x")

        # Row 0 — frame scrubber + playback
        row0 = tk.Frame(ctrl)
        row0.pack(fill="x", pady=2)

        tk.Label(row0, text="Frame:").pack(side="left")
        self.frame_var = tk.IntVar(value=0)
        self.frame_slider = tk.Scale(
            row0, variable=self.frame_var, from_=0, to=1000,
            orient="horizontal", length=500,
            command=self.on_frame_change
        )
        self.frame_slider.pack(side="left", padx=6)
        self.frame_label = tk.Label(row0, text="0 / 0", width=12)
        self.frame_label.pack(side="left")
        tk.Button(row0, text="▶ Play",  command=self.play,  width=8).pack(side="left", padx=4)
        tk.Button(row0, text="⏸ Pause", command=self.pause, width=8).pack(side="left", padx=2)

        # Row 1 — offset controls
        row1 = tk.Frame(ctrl)
        row1.pack(fill="x", pady=2)

        tk.Label(row1, text="Force Offset (s):").pack(side="left")
        self.offset_var = tk.DoubleVar(value=0.0)
        offset_spin = tk.Spinbox(
            row1, textvariable=self.offset_var,
            from_=-9999, to=9999, increment=0.01, width=9
        )
        offset_spin.pack(side="left", padx=6)
        offset_spin.bind("<Return>",   lambda e: self.on_offset_change())
        offset_spin.bind("<FocusOut>", lambda e: self.on_offset_change())

        for label, delta in [("◀◀ -0.1s", -0.1), ("◀ -0.01s", -0.01),
                              ("▶ +0.01s", +0.01), ("▶▶ +0.1s", +0.1)]:
            tk.Button(row1, text=label, command=lambda d=delta: self.nudge(d), width=9).pack(side="left", padx=2)

        tk.Label(row1, text="  positive = shift force data later", fg="gray").pack(side="left", padx=8)

        # Middle: video (fixed width) + force plot (expands)
        content = tk.Frame(self.root)
        content.pack(side="top", fill="both", expand=True, padx=6, pady=4)

        video_frame = tk.LabelFrame(content, text="Video", width=640, height=490)
        video_frame.pack(side="left", fill="both", expand=False, padx=(0, 4))
        video_frame.pack_propagate(False)

        self.video_label = tk.Label(video_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)

        plot_frame = tk.LabelFrame(content, text="Force Data")
        plot_frame.pack(side="left", fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.fig.tight_layout(pad=2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw_empty_plot()

    # ── Helpers ───────────────────────────────────────────────────────────
    def _draw_empty_plot(self):
        self.ax.cla()
        self.ax.set_title("Force Data")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Force")
        self.ax.text(0.5, 0.5, "Load a force data file", transform=self.ax.transAxes,
                     ha="center", va="center", color="gray", fontsize=12)
        self.canvas.draw()

    # ── Dialog-based loaders (manual override buttons) ────────────────────
    def load_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if path:
            self._load_video_from_path(path)

    def load_force(self):
        path = filedialog.askopenfilename(
            filetypes=[("Data files", "*.txt *.csv *.tsv *.xlsx *.xls"), ("All files", "*.*")]
        )
        if path:
            self._load_force_from_path(path)

    def _load_video_from_path(self, path):
        if not path or not os.path.exists(path):
            messagebox.showerror("File Not Found", f"Video file not found:\n{path}")
            return
        if self._owns_cap and self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self._owns_cap = True
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_slider.config(to=max(total - 1, 1))
        self.frame_var.set(0)
        self.root.title(f"Alignment Tool — {os.path.basename(path)}")
        self.show_frame(0)

    def _load_force_from_dataframe(self, df):
        """
        Load directly from an already-parsed DataFrame (self.Force.data).
        Uses the first two numeric columns as (time, force).
        To use a specific column like Fz1, change numeric_cols[1] to df['Fz1'].
        """
        try:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if len(numeric_cols) < 2:
                self.force_time = np.arange(len(df)) / 1000.0
                self.force_data = df[numeric_cols[0]].to_numpy(dtype=float)
            else:
                self.force_time = df[numeric_cols[0]].to_numpy(dtype=float)
                self.force_data = df[numeric_cols[1]].to_numpy(dtype=float)

            # Drop NaN rows
            valid = ~(np.isnan(self.force_time) | np.isnan(self.force_data))
            self.force_time = self.force_time[valid]
            self.force_data = self.force_data[valid]

            self.update_plot()
        except Exception as e:
            messagebox.showerror("Force Data Error", f"Could not read force DataFrame:\n{e}")

    def _load_force_from_path(self, path):
        """Fallback: parse force file from disk using the same formats your app supports."""
        if not path or not os.path.exists(path):
            messagebox.showerror("File Not Found", f"Force file not found:\n{path}")
            return
        try:
            import pandas as pd
            if path.endswith(".xlsx") or path.endswith(".xls"):
                df = pd.read_excel(path)
            elif path.endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = None
                for sep in (",", "\t", r"\s+"):
                    try:
                        candidate = pd.read_csv(path, sep=sep, engine="python")
                        if candidate.shape[1] > 1:
                            df = candidate
                            break
                    except Exception:
                        continue
                if df is None:
                    raise ValueError("Could not parse with comma, tab, or whitespace delimiters.")
            df = df.apply(pd.to_numeric, errors="coerce")
            self._load_force_from_dataframe(df)
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not parse force file:\n{path}\n\n{e}")

    # ── Playback ──────────────────────────────────────────────────────────
    def show_frame(self, frame_idx):
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame = int(frame_idx)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        scale = min(630 / w, 470 / h)
        frame_resized = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))

        photo = ImageTk.PhotoImage(Image.fromarray(frame_resized))
        self.video_label.config(image=photo)
        self.video_label.image = photo  # prevent GC

        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_label.config(text=f"{self.current_frame} / {total - 1}")
        self.update_plot()

    def on_frame_change(self, val):
        self.show_frame(int(float(val)))

    def play(self):
        self.playing = True
        self._play_loop()

    def pause(self):
        self.playing = False

    def _play_loop(self):
        if not self.playing or self.cap is None:
            return
        next_frame = self.current_frame + 1
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if next_frame >= total:
            self.playing = False
            return
        self.frame_var.set(next_frame)
        self.show_frame(next_frame)
        self.root.after(int(1000 / self.video_fps), self._play_loop)

    # ── Alignment ─────────────────────────────────────────────────────────
    def on_offset_change(self):
        try:
            self.offset = float(self.offset_var.get())
        except ValueError:
            pass
        self.update_plot()

    def nudge(self, delta):
        self.offset = round(self.offset + delta, 4)
        self.offset_var.set(self.offset)
        self.update_plot()

    def update_plot(self):
        if self.force_time is None:
            return
        self.ax.cla()
        shifted_time = self.force_time + self.offset
        self.ax.plot(shifted_time, self.force_data, color="steelblue", linewidth=1, label="Force")

        if self.cap is not None:
            t = self.current_frame / self.video_fps
            self.ax.axvline(x=t, color="red", linestyle="--", linewidth=1.5, label=f"Video t={t:.3f}s")

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Force")
        self.ax.set_title(f"Force Data  |  offset = {self.offset:.4f}s")
        self.ax.legend(fontsize=8)
        self.fig.tight_layout(pad=2)
        self.canvas.draw()

    # ── Export ────────────────────────────────────────────────────────────
    def export_alignment(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("CSV", "*.csv")]
        )
        if not path:
            return
        with open(path, "w") as f:
            f.write(f"force_time_offset_seconds={self.offset}\n")
            f.write(f"video_fps={self.video_fps}\n")
            if self.force_time is not None:
                f.write("adjusted_time,force\n")
                for t, v in zip(self.force_time + self.offset, self.force_data):
                    f.write(f"{t:.6f},{v:.6f}\n")
        messagebox.showinfo("Exported", f"Alignment saved to:\n{path}")