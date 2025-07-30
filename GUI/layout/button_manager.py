# GUI/layout/button_manager.py
from callbacks.vector_overlay_combined import vectorOverlayWithAlignmentCallback

import tkinter as tk

class ButtonManager:
    def __init__(self, parent):
        self.parent = parent  # expects a DisplayApp instance
        self.buttons = {}

    def create_buttons(self):
        # self.buttons['align'] = tk.Button(self.parent.master, text="Align", command=self.parent.align)
        self.buttons['graph_option'] = tk.Button(self.parent.master, text="Graphing Options", command=self.parent.graph)
        self.buttons['step_forward'] = tk.Button(self.parent.master, text="+1frame", command=lambda: self.parent.stepF(1))
        self.buttons['step_backward'] = tk.Button(self.parent.master, text="-1frame", command=lambda: self.parent.stepF(-1))
        self.buttons['rotateR'] = tk.Button(self.parent.master, text="Rotate clockwise", command=lambda: self.parent.rotateCam(1))
        self.buttons['rotateL'] = tk.Button(self.parent.master, text="Rotate counterclockwise", command=lambda: self.parent.rotateCam(-1))
        self.buttons['upload_video'] = tk.Button(self.parent.master, text="Upload Video", command=self.parent.upload_video)
        self.buttons['upload_force'] = tk.Button(self.parent.master, text="Upload Force File", command=self.parent.upload_force_data)
        self.buttons['vector_overlay'] = tk.Button(self.parent.master, text="Vector Overlay", command=lambda: vectorOverlayWithAlignmentCallback(self.parent))
        self.buttons['label_video'] = tk.Button(self.parent.master, text="Label Video", command=self.parent.label_video)
        self.buttons['label_force'] = tk.Button(self.parent.master, text="Label Force", command=self.parent.label_force)
        self.buttons['save'] = tk.Button(self.parent.master, text="Save", command=self.parent.save)
        self.buttons['COM'] = tk.Button(self.parent.master, text="COM", command=self.parent.startCOM)

        return self.buttons
