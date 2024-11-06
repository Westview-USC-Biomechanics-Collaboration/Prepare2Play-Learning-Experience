import cv2
import numpy as np
class timeline():
    def __init__(self,rows,frames,force_label=0,video_label=0):
        self.rows = rows
        self.frames = frames

        self.force_label = force_label
        self.video_label = video_label


    def update_force_label(self,num):
        self.force_label = num

    def update_video_label(self,num):
        self.video_label = num


    def create_rect(self):
        # Create a blank image (e.g., 400x400 with a white background)
        image = np.ones((75,1080, 3), dtype="uint8") * 255

        # Define rectangle parameters: top-left and bottom-right coordinates
        top_left = (10,10)
        bottom_right = (1070, 65)
        color = (0, 0, 255)  # Color in BGR (Blue, Green, Red) - here it's red
        thickness = -1  # Thickness of the rectangle border

        # define the trangle label
        vertices = np.array([[10, 15],[0, 5], [20, 5]], np.int32)
        vertices = vertices.reshape((-1, 1, 2))

        # Draw the rectangle
        cv2.rectangle(image, top_left, bottom_right, color, thickness)

        # Draw the filled triangle
        cv2.fillPoly(image, [vertices], (255, 0, 0))

        # Save the image as a PNG file
        cv2.imwrite("rectangle.png", image)

        print("Rectangle image saved as 'rectangle.png'")


if __name__ == "__main__":
    Timeline = timeline(100,100)
    Timeline.create_rect()

