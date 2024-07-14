import pandas as pd


# example import file:
# [[x,y],[a,b]] x,y are the locaiton of forceplate corner in pixel
# []

path_to_forcedata = "data/gis_lr_CC_for02_Raw_Data.csv"
data = pd.read_csv(path_to_forcedata)
example_input = [[457,643],[978,648]]
example_forcedata = [0.000000,	0.566133,	-22.584741,	200.939779,	202.205801,	0.081561,	0.085336,	0.000000,	0.000000,	0.000000,	1.936354,	24.400024,	337.609455,	338.495576,	0.081070,	-0.297077,	0.000000,	0.000000,	0.000000]
index = data.iloc[16]
final_data = pd.DataFrame(data=example_forcedata, index = index)

print(final_data.iloc[5])
# Function:
def calc_angles(location, view):
    import math
    angle_1 = math.atan(float(location[""]))
def find_contact_top(locationin, forcedata):

    import pandas as pd
    force_plate = pd.DataFrame([
        {"plate1" : locationin[0]},
        {"plate2" : locationin[1]},
        {"origin" : [0,0]}
    ])

    calc_angles(force_plate)
















# Call function

find_contact_top(locationin= example_input, forcedata=final_data)