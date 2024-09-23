import math
from test_corners import select_points
import numpy as np

"""
the main() function takes in force data and corner data and RETURNS a new set of start point and end point
"""

def rect_to_trapezoid(x, y, rect_width, rect_height, trapezoid_coords):
    """
    Maps points from a rectangle to a trapezoid, simulating parallax distortion.

    Parameters:
    x, y: Coordinates of the point in the original rectangle (0 <= x <= rect_width, 0 <= y <= rect_height)
    rect_width, rect_height: Dimensions of the original rectangle
    trapezoid_coords: List of four (x, y) tuples representing the trapezoid corners in order:
                      [(top_left), (top_right), (bottom_right), (bottom_left)]

    Returns:
    new_x, new_y: Pixel coordinates of the mapped point in the trapezoid
    """
    # Ensure input coordinates are within the rectangle
    x = np.clip(x, 0, rect_width)
    y = np.clip(y, 0, rect_height)

    # Extract trapezoid coordinates
    (tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y) = trapezoid_coords

    # Calculate the left and right edge positions for the current y
    left_x = tl_x + (bl_x - tl_x) * (1-(y / rect_height))
    right_x = tr_x + (br_x - tr_x) * (y / rect_height)

    # Calculate the width of the trapezoid at the current y
    trapezoid_width = right_x - left_x

    # Map x-coordinate
    new_x = left_x + (x / rect_width) * trapezoid_width

    # Calculate the top and bottom y positions of the trapezoid
    top_y = (tl_y + tr_y) / 2
    bottom_y = (bl_y + br_y) / 2

    # Map y-coordinate
    new_y = top_y + (y / rect_height) * (bottom_y - top_y)

    return (int(new_x), int(new_y))

def normalize_force(force,toTop):
    return force

def main(data,corners):
    # initializing
    fx1 = float(data[0])
    fy1 = float(data[1])
    fz1 = float(data[2])
    px1 = float(data[3]) + 0.45
    py1 = float(data[4]) + 0.3

    fx2 = float(data[5])
    fy2 = float(data[6])
    fz2 = float(data[7])
    px2 = float(data[8]) + 0.45
    py2 = float(data[9]) + 0.3

    first_plate_corners = corners[0:4]
    second_plate_corners = corners[4:]

    start_point1 = rect_to_trapezoid(px1,py1,0.9,0.6,first_plate_corners)
    start_point2 = rect_to_trapezoid(px2,py2,0.9,0.6,second_plate_corners)

    dis_to_top = int((start_point1[1] + start_point2[1])/2)

    fx1 = normalize_force(fx1, dis_to_top)
    fy1 = normalize_force(fy1, dis_to_top)
    fz1 = normalize_force(fz1, dis_to_top)

    fx2 = normalize_force(fx2, dis_to_top)
    fy2 = normalize_force(fy2, dis_to_top)
    fz2 = normalize_force(fz2, dis_to_top)

    # for long view
    end_point1 = (int(start_point1[0]+fy1),int(start_point1[1] - fz1))
    end_point2 = (int(start_point2[0]+fy2),int(start_point2[1] - fz2))

    return ((start_point1,end_point1),(start_point2,end_point2))


if __name__ == "__main__":
    with open("example.in") as fin:
        data = fin.readline().split(" ")

    corner_list = select_points(0, top=True)
    points = main(data, corner_list)
    print(f"points: {points}")


