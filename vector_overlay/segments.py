# Script = segments
# modules: find_head, find_trunk, find_other
# Author:
#   Chase Chen
# Inputs:
#   datain is a pandas dataframe, see more details in Cal_COM.py
#   name is a str
#   segments is a pd dataframe fetch from segdim_deleva.py
#   origin is the name of the segment joint that we use as the start point to calculate segment COM
#   other is the similar thing as origin
#   rowname is the str that I use to access the correct row in the segments dataframe
# Outputs:
#   Evertying is store as object
#
# Dependencies:
#    None
class segments():
    def __init__(self, name, datain,segment):
        self.name = ""
        self.COM = []
        self.mass = 0
        self.data = datain
        self.segment = segment
    def find_head(self):
        nose_x = self.data["nose_x"]
        nose_y = self.data["nose_y"]
        mass = self.segment["massper"]["head"]
        self.mass = float(mass)
        self.COM = [nose_x,nose_y]

    def find_trunk(self):
        # print("This is the trunk method")

        LSHOULDER_x = self.data["left_shoulder_x"]
        LSHOULDER_y = self.data["left_shoulder_y"]
        RSHOULDER_x = self.data["right_shoulder_x"]
        RSHOULDER_y = self.data["right_shoulder_y"]
        LHIP_x = self.data["left_hip_x"]
        LHIP_y = self.data["left_hip_y"]
        RHIP_x = self.data["right_hip_x"]
        RHIP_y = self.data["right_hip_y"]

        SHOULDERMID_x = (LSHOULDER_x+RSHOULDER_x)/2
        SHOULDERMID_y = (LSHOULDER_y+RSHOULDER_y)/2
        # Find midpoints
        HIPMID_x = (LHIP_x+RHIP_x)/2
        HIPMID_y = (LHIP_y+RHIP_y)/2

        COM_x = 0.333*(SHOULDERMID_x - HIPMID_x) + HIPMID_x
        COM_y = 0.333*(SHOULDERMID_y - HIPMID_y) + HIPMID_y
        #assign values
        mass = self.segment["massper"]["trunk"]
        self.mass = float(mass)
        self.COM = [COM_x, COM_y]


    # orignin is the str of the origin joint without the _x, _y
    # other is the str of the other joint without the _x, _y
    # rowname is the segment name (str)

    def find_others(self, origin, other, rowname):
        # print("This is find_others method")

        origin_x = self.data[str(origin)+"_x"]
        origin_y = self.data[str(origin)+"_y"]

        other_x = self.data[str(other) + "_x"]
        other_y = self.data[str(other) + "_y"]
        #calculation
        COM_x = origin_x + (other_x - origin_x) * float(self.segment["cmpos"][str(rowname)])
        COM_y = origin_y + (other_y - origin_y) * float(self.segment["cmpos"][str(rowname)])

        #assign values
        mass = self.segment["massper"][str(rowname)]
        self.mass = mass
        self.COM = [COM_x,COM_y]









