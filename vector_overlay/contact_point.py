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
path_to_forcedata = "C:\\Users\\16199\Documents\GitHub\Prepare2Play-Learning-Experience-3\data\\bcp_lr_CC_for02_Raw_Data.xlsx"
data = pd.read_excel(path_to_forcedata)
example_input = [[457, 643], [978, 648]]
example_forcedata = data.iloc[18]
print(f"This is row 18: {example_forcedata}")
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
    angle_delta = math.atan(y_diff / x_diff)
    print(f"This is x_diff: {x_diff}")
    print(f"This is y_diff: {y_diff}")
    ratio = (math.sqrt(x_diff**2 + y_diff**2))/0.9
    print(f"This is the ratio: {ratio}")
    meter_pixel_ratio = -ratio

    # force pixel ratio
    force_pixel_ratio = 10

    Forceplate1_ON = False
    Forceplate2_ON = False

    def final(angle_delta, meter_pixel_ratio, force_pixel_ratio, forcedata, a, b, fx, fy):
        a1_coords = float(forcedata.iloc[a]) * meter_pixel_ratio
        b1_coords = float(forcedata.iloc[b]) * meter_pixel_ratio
        print(f"This is a1_coords: {a1_coords}")
        print(f"This is b1_coords: {b1_coords}")
        angle_forceplate1 = float(math.atan(b1_coords / a1_coords))
        Fy1 = float(forcedata.iloc[fy]) * force_pixel_ratio * (1) + b1_coords
        Fx1 = float(forcedata.iloc[fx]) * force_pixel_ratio * (1) + a1_coords
        print(f"This is Fy1: {Fy1}")
        print(f"This is Fx1: {Fx1}")
        angle_force1 = float(math.atan(Fy1 / Fx1))

        def find_hypotenuse(x1, y1):
            hypotenuse = math.sqrt(float(x1) ** 2 + float(y1) ** 2)
            return float(hypotenuse)

        HYPOTENUSE1 = find_hypotenuse(x1=a1_coords, y1=b1_coords)
        HYPOTENUSE3 = find_hypotenuse(x1=Fx1, y1=Fy1)

        def find_deltaxy(angle, line):
            deltax = int(line * (math.cos(angle)))
            deltay = int(line * (math.sin(angle)))
            return [deltax, deltay]

        angle_to_use_1 = angle_forceplate1
        vector1_angle = angle_force1
        if angle_delta != None:
            if angle_delta > 0:
                angle_to_use_1 = angle_forceplate1 + angle_delta
                vector1_angle += angle_delta

            elif angle_delta < 0:
                angle_to_use_1 = angle_forceplate1 - angle_delta
                vector1_angle -= angle_delta
            else:
                print("the force plate is parallel to the view")
                angle_to_use_1 = angle_forceplate1

        endpoint1 = find_deltaxy(vector1_angle, HYPOTENUSE3)
        contactpoint1 = find_deltaxy(angle_to_use_1, HYPOTENUSE1)
        return contactpoint1, endpoint1, Fx1, Fy1, angle_to_use_1, vector1_angle, a1_coords, b1_coords
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
    contactpoint1 =None
    contactpoint2 = None
    endpoint1 = None
    endpoint2 = None
    # it should be origin + delta x/y remember to find the pixel to meter ratio
    if forcedata.iloc[6] != float(0):
        Forceplate1_ON = True
        contactpoint1, endpoint1, Fx1, Fy1, angle_to_use_1, vector1_angle, a1_coords, b1_coords = final(angle_delta,meter_pixel_ratio,force_pixel_ratio,forcedata,6,5,2,1)
        print(f"This is contactpoint1: {contactpoint1}")
        contactpoint1 = addlist(contactpoint1, force_plate["plate1"])
        endpoint1 = addlist(endpoint1, force_plate["plate1"])
    if float(forcedata.iloc[15]) != float(0.000000):
        print(f"This is secondforceplate: {forcedata.iloc[15]}")
        Forceplate2_ON = True
        contactpoint2, endpoint2, Fx2, Fy2, angle_to_use_1, vector1_angle, a1_coords, b1_coords = final(angle_delta,meter_pixel_ratio,force_pixel_ratio,forcedata,15,14,10,9)
        print(f"This is contactpoint2")
        contactpoint2 = addlist(contactpoint2, force_plate["plate2"])
        endpoint2 = addlist(endpoint1, force_plate["plate2"])


    print("Forceplate")
    print(force_plate["plate1"])
    print(force_plate["plate2"])
    print(f"This is contactpoint1: {contactpoint1}")
    print(f"This is contactpoint2: {contactpoint2}")
    print(f"This is endpoint1: {endpoint1}")
    print(f"This is endpoint2: {endpoint2}")
    # Decide return format
    return (contactpoint1), (endpoint1), Fx1, Fy1, angle_to_use_1, vector1_angle, a1_coords, b1_coords



# Call function // Testing code
find_contact_top(locationin=example_input, forcedata=example_forcedata)
