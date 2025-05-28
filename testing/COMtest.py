import tkinter as tk
import testing.COM_helper_temp as com_helper
#from Util import COM_helper
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

    def plotCOM(self):
        slider_value = self.slider.get()
        num_lines = COM_helper_temp.count_lines('coord.txt')
        scaled_value = int(slider_value * num_lines / 100)
        x, y = COM_helper_temp.COM_values('coord.txt', scaled_value) #this is a tuple (x, y)
        self.canvas.create_oval(x - 1, y - 1, x + 1, y + 1, fill="black", outline="")


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    app_instance = app(root)
    app_instance.plotCOM()
    
    root.mainloop()
