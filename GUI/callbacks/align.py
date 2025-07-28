import tkinter as tk
import pandas as pd
import numpy as np

def alignCallback(self):
    print("[INFO] aligning...")
    print(f"[DEBUG] {self.state.force_align}, { self.state.video_align}")

    # update the timeline visually
    start, end = self.timelineManager.timeline1.get_start_end()
    try:
        offset = self.state.force_align - self.state.video_align
        newstart = start-offset/self.slider['to']
        newend = end-offset/self.slider['to']
        newlabel = self.timelineManager.timeline1.get_label()-offset/self.slider['to']
        print(f"[INFO] new start percentage: {newstart}\nnew end percentage: {newend}")
        self.timelineManager.timeline1.update_start_end(newstart,newend)
        self.timelineManager.timeline1.update_label(newlabel)

        # debug
        print(f"[DEBUG] offset value: {offset}")
        #check positive or negative offset:
        if(offset>0):
            self.Force.data = self.Force.data.iloc[int(offset*self.state.step_size + self.state.zoom_pos):,:].reset_index(drop=True)

        else:
            nan_rows = pd.DataFrame(np.nan, index=range(int(-offset*self.state.step_size - self.state.zoom_pos)), columns=self.Force.data.columns)
            self.Force.data = pd.concat([nan_rows, self.Force.data], ignore_index=True)  # We are using + because when we have a positive zoom_pos , the number of added rows is offset*step_size - zoom_pos


        # store some output meta data
        self.state.force_start = self.Force.data.iloc[int(self.state.video_align*self.state.step_size),0]

        self.state.zoom_pos = 0
        self.slider.set(0)
        self.state.loc = 0
        self.plot_force_data()
    except TypeError as e:
        self._pop_up("Missing label!!!")
        print("[ERROR] missing label")