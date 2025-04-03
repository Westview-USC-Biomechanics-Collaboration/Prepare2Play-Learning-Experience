import tkinter as tk
from PIL import Image, ImageTk

# Load external image and set as canvas background
def load_background(canvas, file_path):
    image = Image.open(file_path)
    image = image.resize((500, 400))  # Resize if needed
    bg_image = ImageTk.PhotoImage(image)

    # Set image as canvas background
    canvas.bg_image = bg_image  # Keep reference to prevent garbage collection
    canvas.create_image(0, 0, anchor=tk.NW, image=bg_image)

# Create main window
root = tk.Tk()
root.title("Tkinter Canvas with Button")

# Create canvas
canvas = tk.Canvas(root, width=500, height=400)
canvas.pack()

# Load and set background image
load_background(canvas, "GUI/lookBack.jpg")  # Replace with your image path

# Add a button inside the canvas
btn = tk.Button(root, text="Click Me", command=lambda: print("Button Pressed!"))
canvas.create_window(250, 350, window=btn)  # (x, y) position inside the canvas

root.mainloop()
