# Inputs:
#   datain is a pandas series with proper index label
#   sex has to be a string variable, either "m" or "f"
#   author: Chase Chen
#           chase001cz@gamail.com
#
# Outputs:
#    A pd dataframe that contain the x,y coordinate of the COM
#
# Dependencies:
#   pandas
#   segdim_deleva author: Casey Wiens
#                         cwiens32@gmail.com
#   segments author: Chase Chen
def calculateCOM(dataIn, sex):
    import pandas as pd
    from segments import segments
    from segdim_deleva import segmentdim
    data = dataIn
    reference = segmentdim(sex)

    headobject = segments("head",data, reference)
    headobject.find_head()

    trunkobject = segments("trunk", data, reference)
    trunkobject.find_trunk()

    rowname = ["upperarm"  , "forearm" , "hand"       , "thigh" , "shank"  , "foot"  ,]
    origin  = ["LSHOULDER", "LELBOW"  , "LWRIST"     , "LHIP"  , "LKNEE"  , "LHEEL" , "RSHOULDER", "RELBOW"  , "RWRIST"      , "RHIP"  , "RKNEE"  , "RHEEL" ]
    other   = ["LELBOW"   , "LWRIST"  , "left_index" , "LKNEE" , "LANKLE" , "LTOE"  , "RELBOW"   , "RWRIST"  , "right_index" , "RKNEE" , "RANKLE" , "RTOE"]

    count = 0
    other_list = []
    other_name_list = []
    #iterate through the origin list to find segments COM
    for a in range(len(origin)-1):

        if count == 6:
            count =0
        if a <= 5:
            segment_name = rowname[count] + "_L"
            otherobject = segments(segment_name, data, reference)
            otherobject.find_others(origin[a], other[a], rowname[count])
            other_name_list.append(segment_name)
            other_list.append(otherobject)
        if a > 5 :
            segment_name = rowname[count] + "_R"
            otherobject = segments(segment_name, data, reference)
            otherobject.find_others(origin[a], other[a], rowname[count])
            other_name_list.append(segment_name)
            other_list.append(otherobject)
        count += 1

    # formular = Xcom = (m1*x1+ ...)/(m1 + ...)
    Xcom = ((headobject.COM[0]*headobject.mass) + trunkobject.COM[0]*trunkobject.mass + sum(objects.COM[0]*objects.mass for objects in other_list))/ 1
    Ycom = ((headobject.COM[1]*headobject.mass) + trunkobject.COM[1]*trunkobject.mass + sum(objects.COM[1]*objects.mass for objects in other_list))/ 1


    # Now we have every segment of human body as an object, we need a for loop to iterate through all objects
    # And use the COM attribute and mass attribute to calculate the actual COM
    Final_COM = [Xcom,Ycom]

    return [Xcom,Ycom]







#testing code below:

# data = pd.read_excel("outputs/body_landmarks_from_video.xlsx")
# test = data["nose_x"]
# print(test)

# list = calculateCOM("outputs/body_landmarks_from_video.xlsx", "m")
# print("This is the final result")
# print(list)