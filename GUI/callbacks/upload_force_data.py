import tkinter as tk
import threading
from GUI.Timeline import timeline
from PIL import Image, ImageTk
from Util.ballDropDetect import ballDropDetect, forceSpikeDetect
import math
import pandas as pd

def uploadForceDataCallback(self:tk.Tk,):
    def thread_target():
            process(self)  # do the actual loading
            # Once the thread is done, call _plot_force_data from the main thread
            self.master.after(0, self.plot_force_data)

    uploadForceThread = threading.Thread(target=thread_target, daemon=True)
    uploadForceThread.start()



def process(self):
    """
    upload force data to GUI in background thread
    """
    """
    names = ["abs time (s)", "Fx1", "Fy1", "Fz1", "|Ft1|", "Ax1", "Ay1", "COM px1", "COM py1", "COM pz1",
                "Fx2", "Fy2", "Fz2", "|Ft2|", "Ax2", "Ay2", "COM px2", "COM py2", "COM pz2"]
    """
    file_path = tk.filedialog.askopenfilename(title="Select Force Data File",filetypes=[("Excel or CSV Files", "*.xlsx *.xls *.csv *.txt")])
    self.Force.path = file_path
    print(f"Force data uploaded: {file_path}")
    
    if file_path.endswith('.txt'):
        self.Force.data = self.fileReader.readTxt(file_path)

    # support both csv and excel
    if file_path.endswith('.xlsx'):
        self.Force.data = self.fileReader.readExcel(file_path)
    if file_path.endswith('.csv'):
        self.Force.data = self.fileReader.readCsv(file_path)

    self.Force.data = self.Force.data.apply(pd.to_numeric, errors='coerce')
    self.Force.rows = self.Force.data.shape[0]

    if(self.step_size is None):
        self.step_size = 10

    print(f"num of rows: {self.Force.rows}")
    print(f"step size: {self.step_size}")
    self.force_frame = int(self.Force.rows/self.step_size)  # represent num of frames force data can cover

    # self._plot_force_data()

    # Initialize force timeline
    print(f"force frame: {self.force_frame}")
    """
    # create a timeline object, defining end as (num of frame in force_data /  max slider value)
    # Slider value should be updated to frame count when user upload the video file,
    # otherwise we will use the default slider value(100).
    """
    self.timeline1 = timeline(0,self.force_frame/self.slider['to'])
    forceTimeline = Image.fromarray(self.timeline1.draw_rect(loc=self.loc))
    # Resize the image to fit the canvas size
    canvas_width = self.force_timeline.winfo_width()  # Get the width of the canvas
    canvas_height = self.force_timeline.winfo_height()  # Get the height of the canvas

    # Resize the image to match the canvas size using the new resampling method
    newforceTimeline = forceTimeline.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
    self.timeline_image1 = ImageTk.PhotoImage(newforceTimeline)  # create image object that canvas object accept
    self.force_timeline.create_image(0, 0, image=self.timeline_image1, anchor=tk.NW)

    # auto spike deteciton for force plate 2
    targetRow = forceSpikeDetect(self.Force.data)
    print(f"[DEBUG] target row: {targetRow}")
    targetFrame = math.floor(targetRow/self.step_size)
    print(f"[DEBUG] target frame: {targetFrame}")

    # # update timeline
    # self.loc = targetFrame
    # self.label_force()

