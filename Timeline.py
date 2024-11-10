import cv2
import numpy as np

class timeline:
    """
    A timeline representation:
    -10%    0%                          100%      115%
             ___________________________
            |                           |
            |                           |
            |                           |
            -----------------------------
    """

    def __init__(self, start, end):
        # start and end are two percentage values, it can be negative or exceed the canvas
        self.start = start
        self.end = end
        self.label = None

    def get_start_end(self):
        return self.start, self.end

    def get_label(self):
        return self.label

    def update_start_end(self,sIn, eIn):
        self.start = sIn
        self.end = eIn

    def update_label(self, numIn):
        # numIn will be a percentage value relative to global time/frame
        self.label = numIn

    def draw_rect(self, loc):
        # Define the desired background color (BGR format)
        background_color = (61, 98, 122) # Light blue color in (Blue, Green, Red) format

        # Create a blank image (height, width)
        image = np.ones((75, 1080, 3), dtype="uint8") * 255
        image[:] = background_color  # Set the entire image to the background color

        # Convert percentages to x-values
        start_x = int(self.start * 1080)  # Map start percentage to canvas width
        end_x = int(self.end * 1080)      # Map end percentage to canvas width

        # Define rectangle parameters: top-left and bottom-right coordinates
        top_left = (start_x, 10)
        bottom_right = (end_x, 65)

        # Draw the rectangle
        color = (110, 143, 165)  # Red color in BGR format
        thickness = -1  # Fill the rectangle
        cv2.rectangle(image, top_left, bottom_right, color, thickness)

        # Draw the triangle label if it's set
        if self.label is not None:
            label_x = int(self.label * 1080)  # Convert label percentage to x-value
            vertices = np.array([[label_x, 15], [label_x - 10, 5], [label_x + 10, 5]], np.int32)
            vertices = vertices.reshape((-1, 1, 2))
            triangle_color = (250, 229, 200)  # Blue color for the triangle
            cv2.fillPoly(image, pts=[vertices], color=triangle_color)

            # print(self.label ,label_x)

        # Draw the vertical line
        line_color = (200, 230, 250)   # Green color for the line
        loc_x = int(loc * 1080)

        cv2.line(image, pt1=(loc_x, 0), pt2=(loc_x, image.shape[0]), color=line_color, thickness=2)

        # Save the image as a PNG file
        # cv2.imwrite("rectangle.png", image)
        # print("Rectangle image saved as 'rectangle.png'")
        return image


if __name__ == "__main__":
    timeline_obj = timeline(0, 1)
    timeline_obj.update_label(0.5)  # Update label position to 50%
    timeline_obj.draw_rect(loc=0.75)  # Draw the vertical line at 75%
