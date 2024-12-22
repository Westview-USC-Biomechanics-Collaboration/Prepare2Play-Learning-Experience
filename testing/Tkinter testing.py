import tkinter as tk

root = tk.Tk()
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Create canvas and scrollbar
canvas = tk.Canvas(root, width=800, height=600)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)

# Configure canvas
canvas.configure(yscrollcommand=scrollbar.set)

# Grid layout for canvas and scrollbar
canvas.grid(row=0, column=0, sticky="nsew")
scrollbar.grid(row=0, column=1, sticky="ns")

# Create a frame inside canvas to hold the content
frame = tk.Frame(canvas)

# Create window inside canvas to hold the frame
canvas.create_window((0, 0), window=frame, anchor="nw")

# Add content to the frame
for i in range(50):
    label = tk.Label(frame, text=f"Item {i + 1}")
    label.grid(row=i, column=0, padx=5, pady=5)

# Update scroll region when frame size changes
def on_frame_configure(event=None):
    canvas.configure(scrollregion=canvas.bbox("all"))

frame.bind("<Configure>", on_frame_configure)

# Add mouse wheel scrolling
def on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")

canvas.bind_all("<MouseWheel>", on_mousewheel)

root.mainloop()