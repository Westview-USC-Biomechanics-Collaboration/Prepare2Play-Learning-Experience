import math
import pandas as pd


def find_contact_top(locationin, forcedata):
    """
    Find the contact point and force vector endpoint based on input location and force data.

    Parameters:
        locationin (list): A list containing two lists of force plate locations.
        forcedata (pd.Series): The force data as a pandas Series.

    Returns:
        tuple: Contains contact points and endpoints for force vectors.
    """

    def find_hypotenuse(x1, y1):
        """Calculate the hypotenuse given x and y coordinates."""
        return math.sqrt(x1 ** 2 + y1 ** 2)

    def find_deltaxy(angle, line):
        """Calculate the delta x and y given an angle and line length."""
        deltax = int(line * math.cos(angle))
        deltay = -int(line * math.sin(angle))
        return [deltax, deltay]

    def addlist(list1, list2):
        """Add corresponding elements of two lists."""
        if len(list1) != 0:
            return [list1[i] + list2[0][i] for i in range(len(list1))]
        else:
            raise ValueError("Error: Length mismatch")

    force_plate = pd.DataFrame({
        "plate1": [locationin[0]],
        "plate2": [locationin[1]],
        "origin": [[0, 0]]
    })

    # Calculate the angle delta and meter to pixel ratio
    x_diff = float(force_plate["plate2"].iloc[0][0]) - float(force_plate["plate1"].iloc[0][0])
    y_diff = float(force_plate["plate2"].iloc[0][1]) - float(force_plate["plate1"].iloc[0][1])
    angle_delta = -1*math.atan(y_diff / x_diff)
    ratio = math.sqrt(x_diff ** 2 + y_diff ** 2) / 0.9
    meter_pixel_ratio = -ratio

    force_pixel_ratio = 10

    def final(angle_delta, meter_pixel_ratio, force_pixel_ratio, forcedata, a, b, fx, fy):
        a1_coords = (0.45 + float(forcedata.iloc[a])) * meter_pixel_ratio
        b1_coords = (0.3 + float(forcedata.iloc[b])) * meter_pixel_ratio
        angle_forceplate1 = math.atan(b1_coords / a1_coords)
        Fy1 = float(forcedata.iloc[fy]) * force_pixel_ratio + b1_coords
        Fx1 = a1_coords + float(forcedata.iloc[fx]) * force_pixel_ratio
        angle_force1 = math.atan(Fy1 / Fx1)

        HYPOTENUSE1 = find_hypotenuse(a1_coords, b1_coords)
        HYPOTENUSE3 = find_hypotenuse(Fx1, Fy1)

        angle_to_use_1 = angle_forceplate1 + (angle_delta if angle_delta else 0)
        vector1_angle = angle_force1 + (angle_delta if angle_delta else 0)
        # print(f"angle to use: {angle_to_use_1}")
        endpoint1 = find_deltaxy(vector1_angle, HYPOTENUSE3)
        contactpoint1 = find_deltaxy(angle_to_use_1, HYPOTENUSE1)
        return contactpoint1, endpoint1, Fx1, Fy1, angle_to_use_1, vector1_angle, a1_coords, b1_coords

    contactpoint1, contactpoint2 = None, None
    endpoint1, endpoint2 = None, None
    Fx1, Fy1 = None, None
    angle_to_use_1, vector1_angle = None, None
    a1_coords, b1_coords = None, None

    if forcedata.iloc[6] != 0.0:
        contactpoint1, endpoint1, Fx1, Fy1, angle_to_use_1, vector1_angle, a1_coords, b1_coords = final(
            angle_delta, meter_pixel_ratio, force_pixel_ratio, forcedata, 6, 5, 2, 1)
        contactpoint1 = addlist(contactpoint1, force_plate["plate1"])
        endpoint1 = addlist(endpoint1, force_plate["plate1"])

    if forcedata.iloc[15] != 0.0:
        contactpoint2, endpoint2, Fx2, Fy2, angle_to_use_1, vector1_angle, a1_coords, b1_coords = final(
            angle_delta, meter_pixel_ratio, force_pixel_ratio, forcedata, 15, 14, 11, 10)
        contactpoint2 = addlist(contactpoint2, force_plate["plate2"])
        endpoint2 = addlist(endpoint2, force_plate["plate2"])

    return (
    contactpoint1, endpoint1, Fx1, Fy1, angle_to_use_1, vector1_angle, a1_coords, b1_coords, contactpoint2, endpoint2)

# Example input data (uncomment for testing)
# path_to_forcedata = "C:\\Users\\16199\\Documents\\GitHub\\Prepare2Play-Learning-Experience-3\\data\\bcp_lr_CC_for03_Raw_Data.xlsx"
# data = pd.read_excel(path_to_forcedata)
# example_input = [[457, 643], [978, 648]]
# example_forcedata = data.iloc[18]
# find_contact_top(locationin=example_input, forcedata=example_forcedata)


