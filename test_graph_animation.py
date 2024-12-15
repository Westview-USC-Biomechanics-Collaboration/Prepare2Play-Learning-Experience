import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import BytesIO

class test:
    def __init__(self, f, v):
        self.force = f
        self.video = v
        self.x = None
        self.graph_data = None
        self.selected_view = None

    def read(self):
        self.graph_data = pd.read_excel(self.force, skiprows=19)
        names = ["abs time (s)", "Fx1", "Fy1", "Fz1", "|Ft1|", "Ax1", "Ay1", "COM px1", "COM py1", "COM pz1",
                 "Fx2", "Fy2", "Fz2", "|Ft2|", "Ax2", "Ay2", "COM px2", "COM py2", "COM pz2"]
        self.graph_data.columns = names
        self.cam = cv2.VideoCapture(self.video)
        self.x = self.graph_data.iloc[:, 0]
        print(self.graph_data.columns)

    def render_matplotlib_to_cv2(self):
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        time = self.x

        # Force plate 1
        y1 = self.graph_data.loc[:, 'Fy1']
        y2 = self.graph_data.loc[:, 'Fx1']
        # Force plate 2
        y3 = self.graph_data.loc[:, 'Fy2']
        y4 = self.graph_data.loc[:, 'Fx2']

        # Plot for first figure
        line = ax1.axvline(x=1, color='red', linestyle='--', linewidth=1.5)
        line.set_xdata([1])

        ax1.set_title(f"{self.selected_view} Force Time Graph")
        ax1.plot(time, y1, linestyle='-', color='blue', linewidth=0.5, label='Fy1')
        ax1.plot(time, y2, linestyle='-', color='green', linewidth=0.5, label='Fx1')
        ax1.set_xlabel("Time (s.)")
        ax1.set_ylabel("Forces (N.)")
        ax1.legend()

        # Plot for second figure
        ax2.set_title(f"{self.selected_view} Force Time Graph")
        ax2.plot(time, y3, linestyle='-', color='#D34D4D', linewidth=0.5, label='Fy2')
        ax2.plot(time, y4, linestyle='-', color='#008080', linewidth=0.5, label='Fx2')
        ax2.set_xlabel("Time (s.)")
        ax2.set_ylabel("Forces (N.)")
        ax2.legend()

        # Step 2: Save the plot to separate BytesIO objects
        buf1 = BytesIO()
        fig1.savefig(buf1, format='png')
        buf1.seek(0)  # Go to the beginning of the BytesIO object

        buf2 = BytesIO()
        fig2.savefig(buf2, format='png')
        buf2.seek(0)  # Go to the beginning of the BytesIO object

        # Step 3: Convert the BytesIO object to a NumPy array
        image1 = np.asarray(bytearray(buf1.read()), dtype=np.uint8)
        image2 = np.asarray(bytearray(buf2.read()), dtype=np.uint8)

        # Step 4: Decode the byte array to an OpenCV image
        image1 = cv2.imdecode(image1, cv2.IMREAD_COLOR)
        image2 = cv2.imdecode(image2, cv2.IMREAD_COLOR)

        gap = np.full((image1.shape[0], 1920 - image1.shape[1] * 2, 3), 255, dtype=np.uint8)

        graphs = cv2.hconcat([image1, gap, image2])
        # Print the shape of the images
        print(f"Image 1 - Height: {image1.shape[0]}, Width: {image1.shape[1]}")
        print(f"Image 2 - Height: {image2.shape[0]}, Width: {image2.shape[1]}")

        # Show the images using OpenCV
        cv2.imshow('Matplotlib Plot 1', image1)
        cv2.imshow('Matplotlib Plot 2', image2)
        cv2.imshow('Matplotlib Plot 3', graphs)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    haha = test("C:\\Users\\16199\\Desktop\\data\\Chase\\bcp_lr_CC_for02_Raw_Data.xlsx", "C:\\Users\\16199\\Desktop\\data\\Chase\\bcp_lr_CC_vid02.mp4")
    haha.read()
    haha.render_matplotlib_to_cv2()
