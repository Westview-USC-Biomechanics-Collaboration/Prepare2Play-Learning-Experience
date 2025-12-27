import tkinter as tk
from tkinter import filedialog

def open_file_dialog(title="Select file", filetypes=(("All Files", "*.*"),)):
    """
    SAFE Windows 11 file dialog.
    Creates and destroys its own Tk root every time.
    """

    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()

    try:
        path = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes
        )
    finally:
        root.destroy()

    return path

def save_file_dialog(title="Save file", defaultextension="", filetypes=(("All Files", "*.*"),)):
    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    try:
        return filedialog.asksaveasfilename(
            title=title,
            defaultextension=defaultextension,
            filetypes=filetypes
        )
    finally:
        root.destroy()
