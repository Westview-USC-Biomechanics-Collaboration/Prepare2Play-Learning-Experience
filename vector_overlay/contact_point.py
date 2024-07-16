# Script: a script that ca find contact point and force vector end point.
#
# Packages used: math, pandas
#
# Input:
#   forcedata: expected to be pandas series
#   forceplate location: a list containing two lists
#
# Outputs:
#   To Be Determined
#   contact point location: a list with 2 lists inside
#   force vector endpoint: a list with 2 lists inside
#
# Author:
#   Chase Chen
#   chase001cz@gmail.com
#
import pandas as pd

# Example input data
path_to_forcedata = "data/gis_lr_CC_for02_Raw_Data.xlsx"
data = pd.read_excel(path_to_forcedata)
example_input = [[457, 643], [978, 648]]
example_forcedata = [0.000000,	0.566133,	-22.584741,	200.939779,	202.205801,	0.081561,	0.085336,	0.000000,	0.000000,	0.000000,	1.936354,	24.400024,	337.609455,	338.495576,	0.081070,	-0.297077,	0.000000,	0.000000,	0.000000]
index = data.iloc[16]
final_data = pd.DataFrame(data=example_forcedata, index = index)
# print(final_data)
def find_contact_top(locationin, forcedata):
    import math
    import pandas as pd
    force_plate = pd.DataFrame({
        "plate1": [locationin[0]],
        "plate2": [locationin[1]],
        "origin": [[0, 0]]
    })

    # Access coordinates and convert to float
    x_diff = float(force_plate["plate2"].iloc[0][0]) - float(force_plate["plate1"].iloc[0][0])
    y_diff = float(force_plate["plate2"].iloc[0][1]) - float(force_plate["plate1"].iloc[0][1])

    meter_pixel_ratio = -1

    # forceplate 1 coords remember to convert meters to pixels
    a1_coords = float(forcedata.iloc[6]) * meter_pixel_ratio
    b1_coords = float(forcedata.iloc[5]) * meter_pixel_ratio
    print(a1_coords)
    print(b1_coords)
    # forceplate 2 coords
    a2_coords = float(forcedata.iloc[15]) * meter_pixel_ratio
    b2_coords = float(forcedata.iloc[14]) * meter_pixel_ratio

    # Calculate angles
    angle_delta = math.atan(y_diff / x_diff)
    angle_forceplate1 = float(math.atan(b1_coords/a1_coords))
    angle_forceplate2 = float(math.atan(b2_coords/a2_coords))

    # force pixel ratio
    force_pixel_ratio = 10

    # the X and Y direction need to be double check and fixed
    Fy1 = float(forcedata.iloc[1]) * force_pixel_ratio *(-1)  + b1_coords
    Fx1 = float(forcedata.iloc[2]) * force_pixel_ratio *(-1)  + a1_coords

    Fy2 = float(forcedata.iloc[10]) * force_pixel_ratio *(-1) + b2_coords
    Fx2 = float(forcedata.iloc[11]) * force_pixel_ratio *(-1) + a2_coords

    print(Fy1)
    print(Fx1)

    angle_force1 = float(math.atan(Fy1/Fx1))
    angle_force2 = float(math.atan(Fy2/Fx2))
    # Hypotenuse
    def find_hypotenuse(x1,y1):
        hypotenuse = math.sqrt(float(x1)**2 + float(y1)**2)
        return float(hypotenuse)

    angle_to_use_1 = angle_forceplate1
    angle_to_use_2 = angle_forceplate2
    vector1_angle = angle_force1
    vector2_angle = angle_force2
    if angle_delta != None:
        if angle_delta > 0:
            angle_to_use_1 = angle_forceplate1 + angle_delta
            angle_to_use_2 = angle_forceplate2 + angle_delta
            vector1_angle += angle_delta
            vector2_angle += angle_delta

        elif angle_delta < 0:
            angle_to_use_1 = angle_forceplate1 - angle_delta
            angle_to_use_2 = angle_forceplate2 - angle_delta
            vector1_angle -= angle_delta
            vector2_angle -= angle_delta
        else:
            print("the force plate is parallel to the view")
            angle_to_use_1 = angle_forceplate1
            angle_to_use_2 = angle_forceplate2


    HYPOTENUSE1 = find_hypotenuse(x1=a1_coords, y1=b1_coords)
    HYPOTENUSE2 = find_hypotenuse(x1=a2_coords, y1=b2_coords)
    HYPOTENUSE3 = find_hypotenuse(x1=Fx1, y1=Fy1)
    HYPOTENUSE4 = find_hypotenuse(x1=Fx2, y1=Fy2)
    def find_deltaxy(angle, line):
        deltax = line * (math.cos(angle))
        deltay = line * (math.sin(angle))
        return [deltax, deltay]

    # it should be origin + delta x/y remember to find the pixel to meter ratio
    contactpoint1 = find_deltaxy(angle_to_use_1, HYPOTENUSE1)
    contactpoint2 = find_deltaxy(angle_to_use_2, HYPOTENUSE2)

    def addlist(list1,list2):
        if len(list1) != 0:
            newlist = []
            for i in range(len(list1)):
                print(f"This is list2[0][i], i = {i}, {list2[0][i]}")
                item = list1[i] + list2[0][i]
                newlist.append(item)
            return newlist
        else:
            print("Error: Length mismatch")

    contactpoint1 = addlist(contactpoint1, force_plate["plate1"])
    contactpoint2 = addlist(contactpoint2, force_plate["plate2"])
    print(force_plate["plate1"])
    print(force_plate["plate2"])
    print(contactpoint1)
    print(contactpoint2)


    # find force end point
    endpoint1 = find_deltaxy(vector1_angle, HYPOTENUSE3)
    endpoint2 = find_deltaxy(vector2_angle, HYPOTENUSE4)

    endpoint1 = addlist(endpoint1, force_plate["plate1"])
    endpoint2 = addlist(endpoint1, force_plate["plate2"])

    print(endpoint1)
    print(endpoint2)
    # Decide return format
    return



# Call function // Testing code
find_contact_top(locationin=example_input, forcedata=final_data)
