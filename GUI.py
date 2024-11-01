import tkinter as tk
from tkinter import filedialog, Canvas, Label, Scale, Frame, Scrollbar

class DisplayApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Multi-Window Display App")

        # Create a canvas for scrolling
        self.main_canvas = Canvas(master)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a scrollbar
        self.scrollbar = Scrollbar(master, orient="vertical", command=self.main_canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill="y")

        # Configure the canvas to work with the scrollbar
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.main_canvas.bind('<Configure>', lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all")))

        # Create a frame inside the canvas to hold all widgets
        self.frame = Frame(self.main_canvas)
        self.main_canvas.create_window((0, 0), window=self.frame, anchor="nw")

        # Configure weights for columns (0, 1, 2) and rows
        for i in range(3):
            self.master.columnconfigure(i, weight=1)  # Columns for canvases

        # Create three canvases for display in the first row
        self.canvas1 = Canvas(self.frame, width=200, height=150, bg="lightgrey")
        self.canvas1.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.canvas2 = Canvas(self.frame, width=200, height=150, bg="lightgrey")
        self.canvas2.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.canvas3 = Canvas(self.frame, width=200, height=150, bg="lightgrey")
        self.canvas3.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")


        # Create a slider in the middle row
        self.slider = Scale(self.frame, from_=0, to=100, orient="horizontal", label="Adjust Value",
                            command=self.update_slider_value)
        self.slider.grid(row=1, column=0, columnspan=3, padx=5, pady=10, sticky="ew")

        # Label to display slider value
        self.slider_value_label = Label(self.frame, text="Slider Value: 0")
        self.slider_value_label.grid(row=2, column=0, columnspan=3, pady=5)

        # Upload buttons in the bottom row
        self.upload_video_button = tk.Button(self.frame, text="Upload Video", command=self.upload_video)
        self.upload_video_button.grid(row=3, column=0, padx=5, pady=10, sticky="ew")

        self.upload_file_button = tk.Button(self.frame, text="Upload Other File", command=self.upload_file)
        self.upload_file_button.grid(row=3, column=2, padx=5, pady=10, sticky="ew")

        # Vector overlay button
        self.show_vector_overlay = tk.Button(self.frame, text="Vector Overlay", command=lambda: print("Vector overlay clicked"))
        self.show_vector_overlay.grid(row=4, column=0, sticky="ew")

        # Save button
        self.save_button = tk.Button(self.frame, text="Save", command=lambda: print("Save clicked"))
        self.save_button.grid(row=4, column=2, sticky="ew")

        # Force timeline label
        self.force_timeline_label = Label(self.frame, text="Force Timeline (unit = frame)")
        self.force_timeline_label.grid(row=5, column=0, sticky="w")

        # Force timeline
        self.force_timeline = Canvas(self.frame, width=1080, height=75, bg="lightblue")
        self.force_timeline.grid(row=6, column=0, columnspan=3, pady=5)

        # Video timeline label
        self.video_timeline_label = Label(self.frame, text="Video Timeline (unit = frame)")
        self.video_timeline_label.grid(row=7, column=0, sticky="w")

        # Video timeline
        self.video_timeline = Canvas(self.frame, width=1080, height=75, bg="lightblue")
        self.video_timeline.grid(row=8, column=0, columnspan=3, pady=5)

    def update_slider_value(self, value):
        # Update the label with the current slider value
        self.slider_value_label.config(text=f"Slider Value: {value}")

    def upload_video(self):
        # Open a file dialog for video files
        video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
        if video_path:
            print(f"Video uploaded: {video_path}")

    def upload_file(self):
        # Open a file dialog for any file type
        file_path = filedialog.askopenfilename()
        if file_path:
            print(f"File uploaded: {file_path}")


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("600x400")  # Set a default window size
    app = DisplayApp(root)
    root.mainloop()
