import tkinter as tk
import cv2
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from datetime import datetime
import pandas as pd
def saveCallback(self):
    print("user clicked save button")
    file_path = tk.filedialog.asksaveasfilename(
        defaultextension=".mp4",  # Default extension if none is provided
        filetypes=[("MP4 file", "*.mp4"), ("All files", "*.*")]
    )

    def _label(com_in):
        if(com_in<0):# label start
            print("You just labeled start")
            self.save_start = self.save_loc
            self.StartLabel.config(text=f"start frame: {self.save_start}")

        else:
            print("You just labeled end")
            self.save_end = self.save_loc
            self.EndLabel.config(text=f"start frame: {self.save_end}")

    def _scrollBar(value):
        self.save_loc = self.save_scroll_bar.get()
        print(f"You just moved scroll bar to {self.save_loc}")
        self.Video.vector_cam.set(cv2.CAP_PROP_POS_FRAMES, self.save_loc)
        if self.Video.frame_width>self.Video.frame_height:
            self.save_photoImage = self.frameConverter.cvToPillow(camera=self.Video.vector_cam,width=480,height=360)
        else:
            self.save_photoImage = self.frameConverter.cvToPillow(camera=self.Video.vector_cam,width=360,height=480)
        self.save_view_canvas.delete("frame_image")
        self.save_view_canvas.create_image(0, 0, image=self.save_photoImage, anchor=tk.NW, tags="frame_image")

        """
        I notice that when I alter the scroll bar in main window, and then move the scroll bar in toplevel window, the image update
        Possibly because scroll bar in main can also change Video.cam. therefore the solution is to link the two together.
        ### solved
        """
    def _export():
        #matplotlib.use('Agg')
        self._pop_up(text="Processing video ...\nThis may take a few minutes\n",follow=True)

        try:
            print(f"\nentry: {self.cushion_entry.get()}\nfps: {self.Video.fps}")
            cushion_frames = int(self.cushion_entry.get()) * (self.Video.fps)
            print(cushion_frames)
        except ValueError as e:
            self._pop_up(text="Invalid cushion time, please put numbers")
            print("invalid input!!\n")
            print(e)
            return

        # self.Video.cam.set(cv2.CAP_PROP_POS_FRAMES, self.save_start - cushion_frames)
        # self.Video.vector_cam.set(cv2.CAP_PROP_POS_FRAMES, self.save_start - cushion_frames)
        # count = self.save_start - cushion_frames

        print(f"cam1 frame: {self.Video.cam.get(cv2.CAP_PROP_FRAME_COUNT)}\ncam2 frame:{self.Video.vector_cam.get(cv2.CAP_PROP_FRAME_COUNT)}")

        # creating matplot graph
        fig1,ax1 = plt.subplots()
        fig2,ax2 = plt.subplots()
        # time = self.x

        # if(self.selected_view.get()=="Long View"):
        #     label1_1 = "Fy1"
        #     label1_2 = "Fz1"
        #     label2_1 = "Fy2"
        #     label2_2 = "Fx2"
        # elif(self.selected_view.get()=="Short View"):
        #     label1_1 = "Fx1"
        #     label1_2 = "Fz1"
        #     label2_1 = "Fx2"
        #     label2_2 = "Fx2"
        # else: # top view
        #     label1_1 = "Fy1"
        #     label1_2 = "Fx1"
        #     label2_1 = "Fy2"
        #     label2_2 = "Fx2"
        if self.selected_view.get() == "Long View":
            label1_1 = "FP1_Fy"
            label1_2 = "FP1_Fz"
            label2_1 = "FP2_Fy"
            label2_2 = "FP2_Fz"   # <-- you had Fx2 here earlier; long view uses Fz for vertical
        elif self.selected_view.get() == "Short View":
            label1_1 = "FP1_Fx"
            label1_2 = "FP1_Fz"
            label2_1 = "FP2_Fx"
            label2_2 = "FP2_Fz"
        else:  # Top View
            label1_1 = "FP1_Fy"
            label1_2 = "FP1_Fx"
            label2_1 = "FP2_Fy"
            label2_2 = "FP2_Fx"

        dfa = self.state.df_aligned.dropna(subset=["FrameNumber"]).copy()
        dfa["FrameNumber"] = dfa["FrameNumber"].astype(int)

        start_f = int(self.save_start - cushion_frames)
        end_f   = int(self.save_end   + cushion_frames)

        baseF = int(dfa["FrameNumber"].min())

        start_f_abs = int(start_f + baseF) 
        end_f_abs = int(end_f + baseF)

        start_abs = start_f_abs
        end_abs   = end_f_abs

        start_vid = start_abs - baseF
        end_vid   = end_abs   - baseF

        # start_vid = start_abs - baseF
        # end_vid = end_abs - baseF

        print("baseF:", baseF)

        print("df_aligned rows:", len(dfa))
        print("df_aligned FrameNumber range:", dfa["FrameNumber"].min(), dfa["FrameNumber"].max())

        print("absolute export range:", start_abs, end_abs)

        print("requested frame range:", start_f, end_f)

        # Clamp to available aligned range
        minF = int(dfa["FrameNumber"].min())
        maxF = int(dfa["FrameNumber"].max())

        start_f_abs_clamped = max(start_abs, minF)
        end_f_abs_clamped   = min(end_abs, maxF)

        print("clamped ABS frame range:", start_f_abs_clamped, end_f_abs_clamped)

        if start_f_abs_clamped > end_f_abs_clamped:
            raise RuntimeError(
                f"Selected export range (absolute) [{start_abs}, {end_abs}] does not overlap df_aligned "
                f"[{minF}, {maxF}]."
            )


        print("clamped frame range:", start_f_abs_clamped, end_f_abs_clamped)

        dfw = dfa[(dfa["FrameNumber"] >= start_f_abs_clamped) & (dfa["FrameNumber"] <= end_f_abs_clamped)].copy()
        print("dfw rows:", len(dfw))
        if len(dfw) == 0:
            raise RuntimeError("dfw is empty: your save range does not overlap df_aligned FrameNumber range.")
        
        force_cols = [label1_1, label1_2, label2_1, label2_2]  # only these

        print("dfw rows:", len(dfw))
        print("dfw FrameNumber range:", dfw["FrameNumber"].min(), dfw["FrameNumber"].max())
        print("force cols present:", [c for c in force_cols if c in dfw.columns])
        print("non-nan counts:", dfw[force_cols].notna().sum())

        
        # force plate 1
        y1 = dfw.loc[:,label1_1]
        y2 = dfw.loc[:,label1_2]
        # force plate 2
        y3 = dfw.loc[:,label2_1]
        y4 = dfw.loc[:,label2_2]

        ymax = max(y1.max(),y2.max(),y3.max(),y4.max())
        ymin = min(y1.min(),y2.min(),y3.min(),y4.min())

        with open("lag.txt", "r") as f:
            lag = int(f.read().strip())
        print(f"Saving Video the lag value is: {lag}")
        # lag_frames = min(lag, int(self.Video.cam.get(cv2.CAP_PROP_FRAME_COUNT)))  # <- you said lag is in frames
        # lag_rows = int(round(lag_frames * self.state.step_size))

        # dfw.loc[0:self.state.step_size*(self.save_start),:] = np.nan
        # dfw.loc[self.state.step_size*self.save_end:,:] = np.nan
        start_vid = start_abs - baseF
        end_vid   = end_abs   - baseF

        self.Video.cam.set(cv2.CAP_PROP_POS_FRAMES, self.save_start - cushion_frames)
        self.Video.vector_cam.set(cv2.CAP_PROP_POS_FRAMES, self.save_start - cushion_frames)

        count = start_abs          # aligned timeline (matches dfw FrameNumber)
        count_vid = self.save_start - cushion_frames          # actual video timeline (matches cv2 read position) 

        save_start_abs = int(self.save_start + baseF)
        save_end_abs   = int(self.save_end   + baseF)

        dfw.loc[dfw["FrameNumber"] < save_start_abs, force_cols] = np.nan
        dfw.loc[dfw["FrameNumber"] > save_end_abs,   force_cols] = np.nan

        time_col = "abs time (s)" if "abs time (s)" in dfw.columns else "Time(s)"
        time = pd.to_numeric(dfw[time_col], errors="coerce")  # length == len(dfw)

        ax1.clear()
        ax1.set_title(f"Force plate 1 Force Time Graph")
        ax1.set_ylim(ymin, ymax*1.2)
        ax1.plot(time, y1, linestyle='-', color='purple', linewidth=1.5, label="Force horizontal")
        ax1.plot(time, y2, linestyle='-', color='green', linewidth=1.5, label="Force vertical")
        ax1.legend()
        ax1.set_xlabel("Time (s.)")
        ax1.set_ylabel("Forces (N.)")

        # line1 = ax1.axvline(x=dfw.iloc[int(count), 0], color='red', linestyle='--', linewidth=1.5)

        # cur_row = int(count * self.state.step_size) + lag_rows
        # cur_row = np.clip(cur_row, 0, len(self.Force.data)-1)

        ax2.clear()
        ax2.set_title(f"Force plate 2 Force Time Graph")
        ax2.set_ylim(ymin, ymax*1.2)
        ax2.plot(time, y3, linestyle='-', color='orange', linewidth=1.5, label="Force horizontal")
        ax2.plot(time, y4, linestyle='-', color='blue', linewidth=1.5, label="Force vertical")
        ax2.legend()
        ax2.set_xlabel("Time (s.)")
        ax2.set_ylabel("Forces (N.)")

        # line2 = ax2.axvline(x=dfw.iloc[int(count), 0], color='red', linestyle='--', linewidth=1.5)
        r0 = dfw.loc[dfw["FrameNumber"] == count]
        cur_t0 = float(r0.iloc[0][time_col]) if len(r0) else float(time.iloc[0])
        line1 = ax1.axvline(x=cur_t0, color="red", linestyle="--", linewidth=1.5)
        line2 = ax2.axvline(x=cur_t0, color="red", linestyle="--", linewidth=1.5)
        def render_matplotlib_to_cv2(cur):
            cur = np.clip(cur, 9, dfw.shape[0]-1)  # 0609 update: make sure value is in range
            # cur is the row
            # LOCtime = dfw.iloc[int(cur),0]
            # line1.set_xdata([LOCtime])
            # line2.set_xdata([LOCtime])

            LOCtime = float(dfw.iloc[int(cur)][time_col])
            line1.set_xdata([LOCtime])
            line2.set_xdata([LOCtime])

            # Step 2: Save the plot to a BytesIO object
            buf1 = BytesIO()
            fig1.savefig(buf1, format='png')
            buf1.seek(0)  # Go to the beginning of the BytesIO object

            buf2 = BytesIO()
            fig2.savefig(buf2, format='png')
            buf2.seek(0)

            # Step 3: Convert the BytesIO object to a NumPy array
            image1 = np.asarray(bytearray(buf1.read()), dtype=np.uint8)
            image2 = np.asarray(bytearray(buf2.read()), dtype=np.uint8)

            # Step 4: Decode the byte array to an OpenCV image
            image1 = cv2.imdecode(image1, cv2.IMREAD_COLOR)
            image2 = cv2.imdecode(image2, cv2.IMREAD_COLOR)

            print(f"Image dimension: {image1.shape[0],image1.shape[1]}")

            total_width = image1.shape[1] + image2.shape[1]
            total_height = image1.shape[0] + image2.shape[0]
            if total_width > 1920:
                raise ValueError("The combined width of image1 and image2 exceeds 1920 pixels.")

            if self.Video.frame_width > self.Video.frame_height:
                gap_width = (1920 - total_width) // 2  # Ensure integer division
                gap = np.full((image1.shape[0], gap_width, 3), 255,
                                dtype=np.uint8)  # Correct shape for horizontal concat

                return cv2.hconcat([gap, image1, image2, gap])

            else:  # Vertical concatenation
                gap_height = (self.Video.frame_height - total_height) // 2

                # ✅ Correct shape: (gap_height, image1.shape[1], 3)
                gap = np.full((gap_height, image1.shape[1], 3), 255, dtype=np.uint8)

                # Ensure all images have the same width before vconcat
                if image1.shape[1] != image2.shape[1]:
                    target_width = min(image1.shape[1], image2.shape[1])  # Resize to smallest width
                    image1 = cv2.resize(image1, (target_width, image1.shape[0]))
                    image2 = cv2.resize(image2, (target_width, image2.shape[0]))
                    gap = cv2.resize(gap, (target_width, gap.shape[0]))

                return cv2.vconcat([gap, image1, image2, gap])  # Now correctly formatted


        SLOW_FACTOR = 2.0   # or 2.5 for presentations
        out_fps = max(1.0, self.Video.fps / SLOW_FACTOR)
        if self.Video.frame_width > self.Video.frame_height:
            out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), out_fps,(self.Video.frame_width, self.Video.frame_height+480))
        else:
            out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), out_fps,(self.Video.frame_width+640, self.Video.frame_height))
 
        # Saving frame with graph
        while(self.Video.vector_cam.isOpened() and count_vid <= end_f):
            ret1, frame1 = self.Video.cam.read()
            ret3, frame3 = self.Video.vector_cam.read()
            if self.COM_intVar.get()==1:
                frame3 = self.COM_helper.drawFigure(frame3, count_vid)
            if not ret1 or not ret3:
                # if this calls when the frame_number is equal to the total frame count then the stream has just ended
                print(f"Can't read frame at position {count_vid}")
                break

            time_col = "abs time (s)" if "abs time (s)" in dfw.columns else "Time(s)"
            time = dfw[time_col].astype(float)
            r = dfw.loc[dfw["FrameNumber"] == count]
            if len(r):
                cur_t = float(r.iloc[0][time_col])
                line1.set_xdata([cur_t])
                line2.set_xdata([cur_t])
            else:
                line1.set_xdata([np.nan])
                line2.set_xdata([np.nan])

            # graphs = render_matplotlib_to_cv2(int(count * self.state.step_size))  # pass in current row
            
            # render graph based on dfw index for count_abs
            idx = dfw.index[dfw["FrameNumber"] == count]
            graphs = render_matplotlib_to_cv2(int(idx[0]) if len(idx) else 0)

            if(count_vid<self.save_start):
                print("doing ori")
                """
                12/10 notes
                combine graphs horizontally and then combine graphs with video vertically
                export the combined frame, need to test on separate file
                """
                if(self.Video.frame_width > self.Video.frame_height):
                    combined_frame = cv2.vconcat([frame1,graphs])
                else:
                    combined_frame = cv2.hconcat([frame1,graphs])

            elif(count_vid<=self.save_end):
                print("doing vector")
                if (self.Video.frame_width > self.Video.frame_height):
                    combined_frame = cv2.vconcat([frame3,graphs])
                else:
                    combined_frame = cv2.hconcat([frame1, graphs])

            else:
                print("doing ori")
                if (self.Video.frame_width > self.Video.frame_height):
                    combined_frame = cv2.vconcat([frame1, graphs])
                else:
                    combined_frame = cv2.hconcat([frame1, graphs])

            cv2.imshow('Matplotlib Plot', cv2.resize(combined_frame,(960,780)))
            if cv2.waitKey(5) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
            out.write(combined_frame)
            count += 1
            count_vid += 1

        plt.close(fig1)
        plt.close(fig2)
        out.release()
        cv2.destroyAllWindows()
        self._pop_up(text=f"Successfully save vector overlay at {file_path}")
        print(f"Successfully save vector overlay at {file_path}")
        name = file_path.split('/')[-1][:-4]
        with open(f"{file_path[:-4]}.txt","w") as fout:
            fout.write(f"{name}'s metadata\n")
            fout.write(f"Video path: {self.Video.path}\n")
            fout.write(f"Total frame: {self.Video.total_frames}\n")
            fout.write(f"FPS: {self.Video.fps}\n")
            fout.write(f"Video start frame: {self.state.video_align}\n\n")

            fout.write(f"Force data path: {self.Force.path}\n")
            fout.write(f"Force start frame(before align && with out small adjustments): {self.state.force_align}\n")
            fout.write(f"Force start time: {self.state.force_start}\n\n") # using video align because it's position after alignment

            fout.write(f"Cushion time: {self.cushion_entry.get()}\n")
            fout.write(f"Cushion frame: {cushion_frames}\n")  # num of frames before interval of interest and num of frame after if applicable

            fout.write(f"Saving time: {datetime.now()}\n")
            fout.write(f"All rights reserved by Westview PUSD")

    # Creating top level
    self.save_window = tk.Toplevel(self.master)
    self.save_window.title("Save Window")
    self.save_window.geometry("400x800")

    # Freeze the main window
    # self.save_window.grab_set()

    # local variables
    self.save_loc=0

    # layout
    self.save_view_canvas = tk.Canvas(self.save_window,width=400, height=300, bg="lightgrey")
    if(self.Video.frame_width>self.Video.frame_height):
        self.save_photoImage = self.frameConverter.cvToPillow(camera=self.Video.vector_cam,width=480,height=360)
    else:
        self.save_photoImage = self.frameConverter.cvToPillow(camera=self.Video.vector_cam,width=360,height=480)
    self.save_view_canvas.create_image(0, 0, image=self.save_photoImage, anchor=tk.NW, tags="frame_image")
    self.save_view_canvas.grid(row=0,column=0,columnspan=3,sticky="nsew")

    self.save_scroll_bar = tk.Scale(self.save_window, from_=0, to=self.Video.total_frames, orient="horizontal", label="select start and end", command=_scrollBar)
    self.save_scroll_bar.grid(row=1,column=0,columnspan=3,sticky="nsew",pady=10)

    self.StartLabel = tk.Label(self.save_window,text=f"start frame: {self.save_start}")
    self.StartLabel.grid(row=2,column=0,sticky="nsew",padx=10,pady=10)
    self.save_start_button = tk.Button(self.save_window,text="label start",command=lambda:_label(-1))
    self.save_start_button.grid(row=3,column=0,sticky="nsew",padx=10,pady=10)

    self.EndLabel = tk.Label(self.save_window,text=f"end frame: {self.save_end}")
    self.EndLabel.grid(row=2,column=2,sticky="nsew",padx=10,pady=10)
    self.save_end_button = tk.Button(self.save_window,text="label end",command=lambda:_label(1))
    self.save_end_button.grid(row=3,column=2,sticky="nsew",padx=10,pady=10)

    self.cushion_entry = tk.Entry(self.save_window)
    self.cushion_entry.grid(row=4,column=1,padx=10,pady=10,sticky="w")
    self.cushion_label = tk.Label(self.save_window, text = "set cushion time(s) :")
    self.cushion_label.grid(row=4,column=0,padx=10,pady=10)

    self.COM_checkbox = tk.Checkbutton(self.save_window, text="active COM", variable=self.COM_intVar)
    self.COM_checkbox.grid(row=5,column=0,columnspan=3,sticky="nsew",padx=10,pady=10)

    self.save_confirm_button = tk.Button(self.save_window,text="export video",command=_export)
    self.save_confirm_button.grid(row=6,column=0,columnspan=3,sticky="nsew",padx=10,pady=10)

    # lift the save window to the front
    self.save_window.lift()