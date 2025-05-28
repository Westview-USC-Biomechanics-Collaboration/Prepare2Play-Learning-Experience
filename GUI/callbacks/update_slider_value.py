import tkinter as tk
import threading
import cv2

def sliderCallback(self, *value):
    process(self,value)

def process(self,*value):
    try:
        # Update the label with the current slider value
        self.loc = self.slider.get()
        if self.video_data_flag:
            # draw video canvas
            self.Video.cam.set(cv2.CAP_PROP_POS_FRAMES, self.loc)
            self.photo_image1 = self.frameConverter.cvToPillow(camera=self.Video.cam, width=round(self.Video.frame_width * self.zoom_factor1),
                                                    height=round(self.Video.frame_height * self.zoom_factor1))
            self.canvas1.create_image(self.offset_x1, self.offset_y1, image=self.photo_image1, anchor="center")
            # update video timeline
            self._update_video_timeline()

        if self.vector_overlay_flag:
            self.Video.vector_cam.set(cv2.CAP_PROP_POS_FRAMES, self.loc)
            if self.COM_flag:
                _, rawFrame = self.Video.vector_cam.read()
                COMFrame = rawFrame.copy()
                COMFrame = self.COM_helper.drawFigure(COMFrame, self.loc)
                self.photo_image3 = self.frameConverter.cvToPillowFromFrame(COMFrame,
                                                        width=round(self.Video.frame_width * self.zoom_factor3),
                                                        height=round(self.Video.frame_height * self.zoom_factor3))
                self.canvas3.create_image(self.offset_x3, self.offset_y3, image=self.photo_image3, anchor="center")

            else:
                self.photo_image3 = self.frameConverter.cvToPillow(camera=self.Video.vector_cam,
                                                        width=round(self.Video.frame_width * self.zoom_factor3),
                                                        height=round(self.Video.frame_height * self.zoom_factor3))
                self.canvas3.create_image(self.offset_x3, self.offset_y3, image=self.photo_image3, anchor="center")

        if self.save_view_canvas:
            # self.save_loc = self.save_scroll_bar.get()
            print(f"You just moved scroll bar to {self.loc}")
            self.Video.cam.set(cv2.CAP_PROP_POS_FRAMES, self.loc)
            self.save_photoImage = self.frameConverter.cvToPillow(camera=self.Video.cam)
            self.save_view_canvas.delete("frame_image")
            self.save_view_canvas.create_image(0, 0, image=self.save_photoImage, anchor=tk.NW, tags="frame_image")


        if self.force_data_flag :  # somehow self.force_data is not None doesn't work, using Force.rows as compensation
            # draw graph canvas
            # normalized_position = int(value) / (self.slider['to'])
            # x_position = self.ax.get_xlim()[0] + normalized_position * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
            try:
                plate_number = "1" if self.plate.get() == "Force Plate 1" else "2"
                x_position = float(self.Force.data.iloc[int(self.loc * self.step_size + self.zoom_pos),0])
                y_value = float(self.Force.data.loc[int(self.loc * self.step_size + self.zoom_pos),f"{self.option.get()}{plate_number}"])
                self.zoom_baseline.set_xdata([self.Force.data.iloc[self.loc*self.step_size,0]])
                self.line.set_xdata([x_position])
                self.text_label.set_text(f"{self.plate.get()}\n{self.option.get()}: {y_value:.2f}")
                self.figure_canvas.draw()

            except IndexError as e:
                print("force data out of range")

            # update force timeline
            self._update_force_timeline()
    except Exception as e:
        print(f"Error in sliderCallback: {e}")
        # Optionally, you can log the error or handle it in a way that doesn't crash the application
        # For example, you could show a message box or write to a log file
        # tk.messagebox.showerror("Error", f"An error occurred: {e}")
        
