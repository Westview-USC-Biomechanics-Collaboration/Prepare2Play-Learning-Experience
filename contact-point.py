# Script: a script that ca find contact point and force vector end point.
#
# Packages used: math, pandas
#
# Input:
#   forcedata: expected to be pandas series
#   forceplate location: a list containing two lists
#
# Outputs:
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

    a1_coords = float(forcedata.iloc[5])
    b1_coords = float(forcedata.iloc[6])

    a2_coords = float(forcedata.iloc[14])
    b2_coords = float(forcedata.iloc[15])
    # Calculate angles
    angle_delta = math.atan(y_diff / x_diff)
    angle_forceplate1 = float(math.atan(b1_coords/a1_coords))
    angle_forceplate2 = float(math.atan(b2_coords/a2_coords))

    # Hypotenuse
    def find_hypotenuse(x,y):
        hypotenuse = math.sqrt(float(x)**2 + float(y)**2)
        return float(hypotenuse)

    if angle_delta > 0:
        angle_to_use_1 = angle_forceplate1 + angle_delta
        angle_to_use_2 = angle_forceplate2 + angle_delta
        HYPOTENUSE1 = find_hypotenuse(a1_coords,b1_coords)
        HYPOTENUSE2 = find_hypotenuse((a2_coords,b2_coords))

        def find_deltaxy(angle, line):
            deltax = line*(math.cos(angle))
            deltay = line*(math.sin(angle))
            return [deltax,deltay]

        # it should be origin + delta x/y remember to find the pixel to meter ratio
        contactpoint1 = find_deltaxy(angle_to_use_1,HYPOTENUSE1)
        contactpoint2 = find_deltaxy(angle_to_use_2,HYPOTENUSE2)




    elif angle_delta < 0:
        pass
    else:
        pass

# Call function // Testing code
find_contact_top(locationin=example_input, forcedata=final_data)
