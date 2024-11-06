import tkinter as tk
from tkinter import filedialog, Canvas, Label, Scale, Frame, Scrollbar
from PIL import Image, ImageTk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DisplayApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Multi-Window Display App")
        self.master.geometry("1500x800")

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

        # Create three canvases for display in the first row
        self.canvas1 = Canvas(self.frame, width=400, height=300, bg="lightgrey")
        self.canvas1.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.canvas2 = Canvas(self.frame, width=400, height=300, bg="lightgrey")
        self.canvas2.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.canvas3 = Canvas(self.frame, width=400, height=300, bg="lightgrey")
        self.canvas3.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        # Create a slider in the middle row
        self.slider = Scale(self.frame, from_=0, to=100, orient="horizontal", label="Adjust Value",
                            command=self.update_slider_value)
        self.slider.grid(row=1, column=0, columnspan=3, padx=5, pady=10, sticky="ew")

        # Label to display slider value
        self.slider_value_label = Label(self.frame, text="Slider Value: 0")
        self.slider_value_label.grid(row=2, column=0, columnspan=3, pady=5)

        # Upload buttons in the bottom row
        self.upload_csv_button = tk.Button(self.frame, text="Upload .csv", command=self.upload_csv)
        self.upload_csv_button.grid(row=3, column=1, padx=5, pady=10, sticky="ew")

        # Initialize the line reference
        self.line = None

    def update_slider_value(self, value):
        # Update the label with the current slider value
        self.slider_value_label.config(text=f"Slider Value: {value}")

        # Update the line position based on slider value if the line exists
        if self.line:
            max_val = self.slider['to']  # Maximum slider value
            normalized_position = int(value) / max_val
            x_position = self.ax.get_xlim()[0] + normalized_position * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
            self.line.set_xdata([x_position, x_position])
            self.canvas.draw()

    def upload_csv(self):
        # Open a file dialog for CSV files
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                # Load the CSV file starting from row 20, with the first two columns as x and y
                data = pd.read_csv(file_path, skiprows=19, header=None)
                x = data.iloc[:, 0]
                y = data.iloc[:, 1]
                
                # Plot the data
                self.plot_csv_data(x, y)
            except Exception as e:
                print(f"Error loading CSV: {e}")

    def plot_csv_data(self, x, y):
        # Clear previous figure on canvas2
        for widget in self.canvas2.winfo_children():
            widget.destroy()

        # Create a new figure and plot
        self.fig, self.ax = plt.subplots(figsize=(4.75, 3.75))
        self.ax.plot(x, y, marker='o', linestyle='-', color='blue', linewidth = 0.5)
        self.ax.set_title("Force vs. Time")
        self.ax.set_xlabel("Force (N.)")
        self.ax.set_ylabel("Time (s.)")

        # Draw an initial vertical line on the left
        self.line = self.ax.axvline(x=x.iloc[0], color='red', linestyle='--', linewidth=1.5)

        # Embed the matplotlib figure in the Tkinter canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.canvas2)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = DisplayApp(root)
    root.mainloop()
