import os
import sys
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk


class BackgroundManager:
    def __init__(self, master):
        self.master = master
        self.background = None
        self.bg_image = None
        self.bg_image_path = None

        # State variables used elsewhere in your project
        self.plate = tk.StringVar(value="Force Plate 1")
        self.option = tk.StringVar(value="Fz")

    def init_background(self):
        """Initialize background canvas with a cover-style image"""
        # Ensure window is sized before we grab dimensions
        self.master.update_idletasks()
        win_w = self.master.winfo_width()
        win_h = self.master.winfo_height()

        # Determine app path
        if getattr(sys, 'frozen', False):
            app_path = sys._MEIPASS
        else:
            app_path = os.path.dirname(__file__)

        self.bg_image_path = os.path.join(app_path, "lookBack.png")

        # Create the canvas
        self.background = Canvas(self.master, width=win_w, height=win_h)
        self.background.pack(fill=tk.BOTH, expand=True)

        # Load & draw image
        self._draw_background(win_w, win_h)

        # Re-render background on window resize
        self.master.bind("<Configure>", self._on_resize)

        return self.background

    def _draw_background(self, width, height):
        """Helper: resize and draw background image with aspect ratio preserved"""
        try:
            image = Image.open(self.bg_image_path)

            # Cover-mode resize
            ratio = max(width / image.width, height / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.LANCZOS)

            # Crop to window size (center crop)
            left = (image.width - width) // 2
            top = (image.height - height) // 2
            image = image.crop((left, top, left + width, top + height))

            # Convert to PhotoImage
            self.bg_image = ImageTk.PhotoImage(image)

            # Draw image
            # self.background.delete("bg")  # clear old
            self.background.config(width=width, height=height)
            self.background.create_image(0, 0, image=self.bg_image, anchor="nw", tag="bg")
            self.background.tag_lower("bg")

        except FileNotFoundError:
            print(f"[ERROR] Background image not found: {self.bg_image_path}")
            self.background.config(bg="gray")

    def _on_resize(self, event):
        """Callback when the window is resized"""
        if event.width > 1 and event.height > 1:  # prevent tiny invalid calls
            self._draw_background(event.width, event.height)
