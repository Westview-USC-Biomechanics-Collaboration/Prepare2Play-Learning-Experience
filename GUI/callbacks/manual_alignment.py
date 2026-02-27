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
        self.force_fp1  = None          # FP1 vertical force (Fz)
        self.force_fp2  = None          # FP2 vertical force (Fz)
        self.force_df   = None          # full force DataFrame for column lookup
        self.offset = 0.0
        self._vline       = None        # matplotlib vline reference for fast update
        self._vline_label = None        # legend entry for the vline
        self.selected_plate = None      # set in _build_ui
        self.selected_component = None  # set in _build_ui
        self.video_fps = 30
        self.current_frame = 0
        self.playing = False

        self._build_ui()

        # ── Auto-load video ───────────────────────────────────────────────
        if video is not None:
            self._init_video(video)

        # ── Auto-load force ───────────────────────────────────────────────
        if force is not None:
            self._init_force(force)

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

        if hasattr(video, "path") and video.path:
            self.root.title(f"Alignment Tool — {os.path.basename(video.path)}")

        # Defer first frame render until the window is fully realized.
        # Creating ImageTk.PhotoImage before the window exists causes
        # "pyimageN doesn't exist" TclError.
        self.root.after(100, lambda: self.show_frame(0))

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
        tk.Button(btn_frame, text="✓ Confirm & Close", command=self.export_alignment, width=16).pack(side="right", padx=4)

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
        tk.Button(row0, text="◀ -1",   command=lambda: self.step_frame(-1), width=5).pack(side="left", padx=2)
        tk.Button(row0, text="▶ Play",  command=self.play,  width=8).pack(side="left", padx=2)
        tk.Button(row0, text="⏸ Pause", command=self.pause, width=8).pack(side="left", padx=2)
        tk.Button(row0, text="+1 ▶",   command=lambda: self.step_frame(+1), width=5).pack(side="left", padx=2)

        # Row 1 — offset slider (fast coarse adjustment)
        row1 = tk.Frame(ctrl)
        row1.pack(fill="x", pady=2)

        tk.Label(row1, text="Force Offset (s):").pack(side="left")
        self.offset_var = tk.DoubleVar(value=0.0)

        self.offset_range = tk.IntVar(value=30)
        self.offset_slider = tk.Scale(
            row1, variable=self.offset_var, from_=-30, to=30,
            orient="horizontal", resolution=0.01, length=400,
            command=lambda v: self.on_offset_change()
        )
        self.offset_slider.pack(side="left", padx=6)

        # Range selector — widens/narrows the slider range
        tk.Label(row1, text="Range:").pack(side="left")
        for r in (5, 30, 120):
            tk.Button(row1, text=f"±{r}s", width=5,
                      command=lambda v=r: self._set_offset_range(v)).pack(side="left", padx=1)

        # Row 2 — fine-tune spinbox + nudge buttons
        row2 = tk.Frame(ctrl)
        row2.pack(fill="x", pady=2)

        tk.Label(row2, text="Fine tune:").pack(side="left")
        offset_spin = tk.Spinbox(
            row2, textvariable=self.offset_var,
            from_=-9999, to=9999, increment=0.01, width=9
        )
        offset_spin.pack(side="left", padx=6)
        offset_spin.bind("<Return>",   lambda e: self.on_offset_change())
        offset_spin.bind("<FocusOut>", lambda e: self.on_offset_change())

        for label, delta in [("◀◀ -0.1s", -0.1), ("◀ -0.01s", -0.01),
                              ("▶ +0.01s", +0.01), ("▶▶ +0.1s", +0.1)]:
            tk.Button(row2, text=label, command=lambda d=delta: self.nudge(d), width=9).pack(side="left", padx=2)

        tk.Label(row2, text="  positive = shift force data later", fg="gray").pack(side="left", padx=8)

        # Row 3 — graph channel selector (matches graphOptionCallback style)
        row3 = tk.Frame(ctrl)
        row3.pack(fill="x", pady=2)

        tk.Label(row3, text="Force Plate:").pack(side="left")
        self.selected_plate = tk.StringVar(value="Both")
        for text, val in [("Both", "Both"), ("FP1", "Force Plate 1"), ("FP2", "Force Plate 2")]:
            tk.Radiobutton(row3, text=text, variable=self.selected_plate, value=val,
                           command=self.update_plot).pack(side="left", padx=4)

        tk.Label(row3, text="   Component:").pack(side="left")
        self.selected_component = tk.StringVar(value="Fz")
        for comp in ("Fx", "Fy", "Fz", "Ax", "Ay"):
            tk.Radiobutton(row3, text=comp, variable=self.selected_component, value=comp,
                           command=self.update_plot).pack(side="left", padx=3)

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
        Extract time, FP1 Fz, and FP2 Fz from the force DataFrame.

        Supports both raw column names from the BioWare text file
        (Fz at position index 3, Fz.1 at position index 9) and the
        already-renamed FP1_Fz / FP2_Fz names used elsewhere in the pipeline.
        """
        try:
            # ── Time axis ────────────────────────────────────────────────
            time_candidates = ['abs time (s)', 'Time(s)', 'time', 'Time']
            time_col = next((c for c in time_candidates if c in df.columns), None)
            if time_col:
                self.force_time = df[time_col].to_numpy(dtype=float)
            else:
                # Fall back to row index converted to seconds at 1200 Hz
                self.force_time = np.arange(len(df)) / 1200.0

            # ── FP1 vertical force (Fz) ──────────────────────────────────
            fp1_candidates = ['Fz1', 'FP1_Fz', 'Fz']
            fp1_col = next((c for c in fp1_candidates if c in df.columns), None)
            self.force_fp1 = df[fp1_col].to_numpy(dtype=float) if fp1_col else np.zeros(len(df))

            # ── FP2 vertical force (Fz) ──────────────────────────────────
            fp2_candidates = ['Fz2', 'FP2_Fz', 'Fz.1']
            fp2_col = next((c for c in fp2_candidates if c in df.columns), None)
            self.force_fp2 = df[fp2_col].to_numpy(dtype=float) if fp2_col else np.zeros(len(df))

            # ── Drop rows where time is NaN ───────────────────────────────
            valid = ~np.isnan(self.force_time)
            df_valid = df.reset_index(drop=True).loc[valid].reset_index(drop=True)
            self.force_time = self.force_time[valid]
            self.force_fp1  = self.force_fp1[valid]
            self.force_fp2  = self.force_fp2[valid]

            # ── Subsample to match video fps (1200 Hz / 120 fps = every 10th row)
            # This means 1 force sample = 1 video frame, so the red line
            # moves exactly in sync with the force data during playback.
            force_hz  = 1200.0
            video_fps = self.video_fps if self.video_fps else 120.0
            step = max(1, round(force_hz / video_fps))  # = 10 for 120fps video
            self.force_time = self.force_time[::step]
            self.force_fp1  = self.force_fp1[::step]
            self.force_fp2  = self.force_fp2[::step]
            self.force_df   = df_valid.iloc[::step].reset_index(drop=True)

            # Keep force_data pointing to FP1 for backward compatibility
            self.force_data = self.force_fp1

            print(f"[AlignmentGUI] force_hz={force_hz}, video_fps={video_fps}, step={step}")
            print(f"[AlignmentGUI] force samples after subsample: {len(self.force_time)}")
            print(f"[AlignmentGUI] force_df columns: {list(self.force_df.columns)}")

            self.update_plot()
        except Exception as e:
            import traceback
            traceback.print_exc()
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
    def _render_video_frame(self, frame_idx):
        """Read and display a video frame. Returns True on success."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = self.cap.read()
        if not ret:
            return False
        self.current_frame = int(frame_idx)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        scale = min(630 / w, 470 / h)
        frame_resized = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))
        photo = ImageTk.PhotoImage(Image.fromarray(frame_resized))
        self.video_label.config(image=photo)
        self.video_label.image = photo
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_label.config(text=f"{self.current_frame} / {total - 1}")
        return True

    def _update_vline(self):
        """During playback, just move the red vertical line without full redraw."""
        if self.force_time is None or not hasattr(self, "_vline"):
            return
        t = self.current_frame / self.video_fps
        if self._vline is not None:
            self._vline.set_xdata([t, t])
            self._vline_label.set_text(f"Video t={t:.3f}s")
            self.canvas.draw_idle()  # lightweight — only redraws what changed

    def show_frame(self, frame_idx):
        """Full update: render video frame + redraw entire plot."""
        if self.cap is None:
            return
        self._render_video_frame(frame_idx)
        self.update_plot()  # full redraw including vline

    def _show_frame_fast(self, frame_idx):
        """Fast update for playback: render video frame + move vline only."""
        if self.cap is None:
            return
        self._render_video_frame(frame_idx)
        self._update_vline()

    def on_frame_change(self, val):
        self.show_frame(int(float(val)))

    def play(self):
        self.playing = True
        self._play_loop()

    def pause(self):
        self.playing = False

    def step_frame(self, delta):
        """Step forward or backward by delta frames."""
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap else 1
        new_frame = max(0, min(self.current_frame + delta, total - 1))
        self.frame_var.set(new_frame)
        self.show_frame(new_frame)

    def _play_loop(self):
        if not self.playing or self.cap is None:
            return
        next_frame = self.current_frame + 1
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if next_frame >= total:
            self.playing = False
            return
        self.frame_var.set(next_frame)
        # During playback only move the vline — skip full redraw for performance
        self._show_frame_fast(next_frame)
        self.root.after(int(1000 / self.video_fps), self._play_loop)

    # ── Alignment ─────────────────────────────────────────────────────────
    def _set_offset_range(self, r):
        """Widen or narrow the offset slider range without losing the current value."""
        self.offset_slider.config(from_=-r, to=r)

    def on_offset_change(self):
        try:
            self.offset = float(self.offset_var.get())
        except ValueError:
            pass
        self.update_plot()

    def nudge(self, delta):
        self.offset = round(self.offset + delta, 4)
        self.offset_var.set(self.offset)  # updates both spinbox and slider
        self.update_plot()

    def _get_force_column(self, plate_label, component):
        """
        Return a numpy array for the requested plate + component combination.
        Checks renamed pipeline names (FP1_Fz) then raw BioWare names (Fz, Fz.1).
        force_df is already filtered to the same valid rows as force_time.
        """
        if self.force_df is None:
            print(f"[_get_force_column] force_df is None")
            return None

        # Map (plate, component) → candidate column names in priority order
        # Covers: Fz1/Fz2 style (your app), FP1_Fz/FP2_Fz (pipeline renamed), Fz/Fz.1 (raw BioWare)
        col_map = {
            ("Force Plate 1", "Fx"): ["Fx2", "FP2_Fx", "Fx.1"],
            ("Force Plate 1", "Fy"): ["Fy2", "FP2_Fy", "Fy.1"],
            ("Force Plate 1", "Fz"): ["Fz2", "FP2_Fz", "Fz.1"],
            ("Force Plate 1", "Ax"): ["Ax2", "FP2_Ax", "Ax.1"],
            ("Force Plate 1", "Ay"): ["Ay2", "FP2_Ay", "Ay.1"],
            ("Force Plate 2", "Fx"): ["Fx1", "FP1_Fx", "Fx"],
            ("Force Plate 2", "Fy"): ["Fy1", "FP1_Fy", "Fy"],
            ("Force Plate 2", "Fz"): ["Fz1", "FP1_Fz", "Fz"],
            ("Force Plate 2", "Ax"): ["Ax1", "FP1_Ax", "Ax"],
            ("Force Plate 2", "Ay"): ["Ay1", "FP1_Ay", "Ay"],
        }
        candidates = col_map.get((plate_label, component), [])
        # print(f"[_get_force_column] plate={plate_label}, comp={component}, trying: {candidates}")
        # print(f"[_get_force_column] available cols: {list(self.force_df.columns)}")
        for col in candidates:
            if col in self.force_df.columns:
                # print(f"[_get_force_column] found column: {col}")
                return self.force_df[col].to_numpy(dtype=float)
        # print(f"[_get_force_column] NO matching column found")
        return None

    def update_plot(self):
        if self.force_time is None:
            return

        self.ax.cla()
        shifted_time = self.force_time + self.offset

        plate_sel = self.selected_plate.get() if self.selected_plate else "Both"
        comp_sel  = self.selected_component.get() if self.selected_component else "Fz"

        colors = {"Force Plate 1": "steelblue", "Force Plate 2": "darkorange"}

        if plate_sel == "Both":
            plates = ["Force Plate 1", "Force Plate 2"]
        else:
            plates = [plate_sel]

        for plate in plates:
            arr = self._get_force_column(plate, comp_sel)
            if arr is not None:
                label = f"FP1 {comp_sel}" if plate == "Force Plate 1" else f"FP2 {comp_sel}"
                self.ax.plot(shifted_time, arr, color=colors[plate], linewidth=1, label=label)

        # Red vertical line at current video timestamp — store reference for fast updates
        if self.cap is not None:
            t = self.current_frame / self.video_fps
            self._vline = self.ax.axvline(x=t, color="red", linestyle="--", linewidth=1.5, label=f"Video t={t:.3f}s")
            # Keep a handle to the legend text so _update_vline can update it
            self._vline_label = self._vline

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel(f"{comp_sel} (N or m)")
        self.ax.set_title(f"{plate_sel} — {comp_sel}  |  offset = {self.offset:.4f}s")
        self.ax.legend(fontsize=8)
        self.fig.tight_layout(pad=2)
        self.canvas.draw()

    # ── Export ────────────────────────────────────────────────────────────
    def export_alignment(self):
        """Close the window — offset is read from app.offset by the caller."""
        self.playing = False
        self.root.destroy()