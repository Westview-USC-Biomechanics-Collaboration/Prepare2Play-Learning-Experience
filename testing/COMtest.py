import tkinter as tk

class app:
    def __init__(self, root: tk.Tk):
        self.master = root
        self.master.title("Tkinter App")
        self.setupUI()
        self.testVar = "let's goooooo"

    def setupUI(self):
        self.canvas = tk.Canvas(self.master, width=800, height=600)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.button = tk.Button(self.canvas, text="Click Me", command=lambda: print("hahaha"))
        self.button.pack(side=tk.BOTTOM, fill=tk.X)
        self.slider = tk.Scale(self.canvas, from_=0, to=100, orient=tk.HORIZONTAL)
        self.slider.pack(side=tk.BOTTOM, fill=tk.X)
        self.subCanvas = tk.Canvas(self.canvas, width=400, height=400,bg="blue")
        self.subCanvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)


    
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    app_instance = app(root)
    root.mainloop()
