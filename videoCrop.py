import cv2
import tkinter as tk
from tkinter import filedialog

class VideoCropper:
    def __init__(self, root, video_path):
        self.root = root
        self.root.title("Video Cropper")

        self.video_path = video_path
        self.video_clip = cv2.VideoCapture(self.video_path)
        self.frame_width = int(self.video_clip.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_clip.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.canvas = tk.Canvas(self.root, width=self.frame_width, height=self.frame_height)
        self.canvas.pack()

        self.load_next_frame()

        self.crop_start = None
        self.crop_end = None

        self.canvas.bind("<ButtonPress-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

    def load_next_frame(self):
        ret, frame = self.video_clip.read()
        if ret:
            self.photo = tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        else:
            self.video_clip.release()

    def on_canvas_click(self, event):
        self.crop_start = (event.x, event.y)

    def on_canvas_drag(self, event):
        self.crop_end = (event.x, event.y)
        self.canvas.delete("rect")
        self.canvas.create_rectangle(self.crop_start[0], self.crop_start[1], self.crop_end[0], self.crop_end[1], outline='red', tag="rect")

    def on_canvas_release(self, event):
        self.crop_end = (event.x, event.y)
        self.canvas.delete("rect")
        self.crop_video()

    def crop_video(self):
        x1 = min(self.crop_start[0], self.crop_end[0])
        y1 = min(self.crop_start[1], self.crop_end[1])
        x2 = max(self.crop_start[0], self.crop_end[0])
        y2 = max(self.crop_start[1], self.crop_end[1])

        success, frame = self.video_clip.read()
        if success:
            cropped_frame = frame[y1:y2, x1:x2]
            cv2.imshow("Cropped Video", cropped_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error reading video frame")

if __name__ == '__main__':
    root = tk.Tk()

    video_path = 'data:derenBasketballTest1.mp4'

    app = VideoCropper(root, video_path)
    root.mainloop()
