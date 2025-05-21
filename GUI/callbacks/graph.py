import tkinter as tk

def graphOptionCallback(self):
    # Create a new popup window
        popup = tk.Toplevel(self.master)
        popup.title("Force Plate Selection")
        popup.geometry("300x250")

        # Variables to store selected radio button values
        self.plate = tk.StringVar(value="Force Plate 1")
        self.option = tk.StringVar(value="Fx")

        # First row: Force Plate Selection
        frame1 = tk.Frame(popup)
        frame1.pack(pady=10)

        tk.Label(frame1, text="Select Force Plate:").pack(side=tk.LEFT)
        force_plate_1 = tk.Radiobutton(frame1, text="Force Plate 1", variable=self.plate,
                                       value="Force Plate 1")
        force_plate_1.pack(side=tk.LEFT)

        force_plate_2 = tk.Radiobutton(frame1, text="Force Plate 2", variable=self.plate,
                                       value="Force Plate 2")
        force_plate_2.pack(side=tk.LEFT)

        # Second row: Force Components
        frame2 = tk.Frame(popup)
        frame2.pack(pady=10)

        tk.Label(frame2, text="Select Force").pack()

        fx_radio = tk.Radiobutton(frame2, text="Fx", variable=self.option, value="Fx")
        fx_radio.pack(side=tk.LEFT, padx=5)

        fy_radio = tk.Radiobutton(frame2, text="Fy", variable=self.option, value="Fy")
        fy_radio.pack(side=tk.LEFT, padx=5)

        fz_radio = tk.Radiobutton(frame2, text="Fz", variable=self.option, value="Fz")
        fz_radio.pack(side=tk.LEFT, padx=5)
        
        px_radio = tk.Radiobutton(frame2, text="Ax", variable=self.option, value="Ax")
        px_radio.pack(side=tk.LEFT, padx=5)

        py_radio = tk.Radiobutton(frame2, text="Ay", variable=self.option, value="Ay")
        py_radio.pack(side=tk.LEFT, padx=5)
        def make_changes():
            try:
                if(self.loc>self.force_frame):
                    self.slider.set(0)
                    self.loc = 0

                self.plot_force_data()
            except AttributeError as e:
                print("Missing force data !!!")
            popup.destroy()

        # Button to confirm and close the popup
        confirm_btn = tk.Button(popup, text="Confirm", command=make_changes)
        confirm_btn.pack(pady=10)