import tkinter as tk
from tkinter import Canvas
from PIL import ImageTk
import cv2

class CanvasManager:
    def __init__(self, parent):
        self.parent = parent  # This is DisplayApp instance
        self.canvas1 = None
        self.canvas2 = None
        self.canvas3 = None
        self.photo_image1 = None
        self.photo_image3 = None

        # Zoom state
        self.zoom_factor1 = 1.0
        self.zoom_factor3 = 1.0
        self.offset_x1 = 200
        self.offset_y1 = 150
        self.offset_x3 = 200
        self.offset_y3 = 150
        self.placeloc1 = [0, 0]
        self.placeloc3 = [0, 0]

    def init_canvases(self):
        canvas_width = 400
        self.canvas1 = Canvas(self.parent.master, width=canvas_width, height=300, bg="lightgrey")
        self.canvas2 = Canvas(self.parent.master, width=canvas_width, height=300, bg="lightgrey")
        self.canvas3 = Canvas(self.parent.master, width=canvas_width, height=300, bg="lightgrey")

        self._bind_canvas_events()
        return self.canvas1, self.canvas2, self.canvas3

    def _bind_canvas_events(self):
        self.canvas1.bind("<ButtonPress-1>", lambda event: self._on_drag(event, canvas=1))
        self.canvas1.bind("<B1-Motion>", lambda event: self._on_drag(event, canvas=1))
        self.canvas1.bind("<ButtonRelease-1>", lambda event: self._on_drag(event, canvas=1))
        self.canvas1.bind("<MouseWheel>", lambda event: self._on_zoom(event, canvas=1))

        self.canvas3.bind("<ButtonPress-1>", lambda event: self._on_drag(event, canvas=3))
        self.canvas3.bind("<B1-Motion>", lambda event: self._on_drag(event, canvas=3))
        self.canvas3.bind("<ButtonRelease-1>", lambda event: self._on_drag(event, canvas=3))
        self.canvas3.bind("<MouseWheel>", lambda event: self._on_zoom(event, canvas=3))

    def _on_zoom(self, event, canvas):
        zoom_factor_attr = f"zoom_factor{canvas}"
        zoom_factor = getattr(self, zoom_factor_attr)

        if event.delta > 0:
            zoom_factor *= 1.1
        else:
            zoom_factor *= 0.9

        zoom_factor = max(0.1, min(zoom_factor, 5.0))
        setattr(self, zoom_factor_attr, zoom_factor)

        if canvas == 1:
            self._redraw_canvas(canvas, self.parent.Video.cam, zoom_factor, frame_number=self.parent.state.loc)
        elif canvas == 3:
            self._redraw_canvas(canvas, self.parent.Video.vector_cam, zoom_factor, frame_number=self.parent.state.loc)

    def _on_drag(self, event, canvas):
        place_attr = f"placeloc{canvas}"
        offset_x_attr = f"offset_x{canvas}"
        offset_y_attr = f"offset_y{canvas}"

        if event.type == "4":  # ButtonPress
            setattr(self, place_attr, [event.x, event.y])

        elif event.type == "6":  # Dragging
            placeloc = getattr(self, place_attr)
            setattr(self, offset_x_attr, getattr(self, offset_x_attr) + (event.x - placeloc[0]))
            setattr(self, offset_y_attr, getattr(self, offset_y_attr) + (event.y - placeloc[1]))
            setattr(self, place_attr, [event.x, event.y])

            camera = self.parent.Video.cam if canvas == 1 else self.parent.Video.vector_cam
            zoom = getattr(self, f"zoom_factor{canvas}")
            self._redraw_canvas(canvas, camera, zoom, frame_number=self.parent.state.loc)

        elif event.type == "5":  # Release
            camera = self.parent.Video.cam if canvas == 1 else self.parent.Video.vector_cam
            zoom = getattr(self, f"zoom_factor{canvas}")
            self._redraw_canvas(canvas, camera, zoom, frame_number=self.parent.state.loc)

    def _redraw_canvas(self, canvas, camera, zoom_factor, frame_number=None):
        canvas_obj = getattr(self, f"canvas{canvas}")
        width = round(self.parent.Video.frame_width * zoom_factor)
        height = round(self.parent.Video.frame_height * zoom_factor)

        photo_image = self.parent.frameConverter.cvToPillow(
            camera=camera,
            width=width,
            height=height,
            frame_number=frame_number  # Freeze at specified frame
        )

        if photo_image is None:
            return  # Handle blank or bad frame

        setattr(self, f"photo_image{canvas}", photo_image)

        canvas_obj.delete("all")
        canvas_obj.create_image(
            getattr(self, f"offset_x{canvas}"),
            getattr(self, f"offset_y{canvas}"),
            image=photo_image,
            anchor="center"
        )
