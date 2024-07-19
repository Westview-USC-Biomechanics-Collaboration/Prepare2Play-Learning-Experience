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

    def round_list(lst):
        return [int(item) for item in lst]

    # Calculate the meter to pixel ratio, also the force_pixel ratio
    forceplate1_corner = locationin[0]
    forceplate2_corner = locationin[1]
    x_diff = float(forceplate2_corner[0] - forceplate1_corner[0])
    meter_pixel_ratio = x_diff/ 0.9
    # print(f"This is meter_pixel_ratio: {meter_pixel_ratio}")
    force_pixel_ratio = 10

    def find_contact_end(cornerlocation, Ax, Ay, fx, fy):
        # find center location in pixel
        center_location = [cornerlocation[0] + 0.5 * x_diff, cornerlocation[1] - x_diff / 3]
        print(f"Center location: {center_location}")
        contactpoint = [center_location[0] + float(forcedata.iloc[Ay]) * meter_pixel_ratio,
                        center_location[1] - float(forcedata.iloc[Ax]) * meter_pixel_ratio]
        # The direction of Ax and Ay need to be fixed
        endpoint = [contactpoint[0] - float(forcedata.iloc[fy]) * force_pixel_ratio,
                    contactpoint[1] + float(forcedata.iloc[fx]) * force_pixel_ratio]

        return contactpoint, endpoint, center_location

    contactpoint1, endpoint1, center1 = find_contact_end(forceplate1_corner,5,6,1,2)
    contactpoint2, endpoint2, center2 = find_contact_end(forceplate2_corner, 14, 15, 10, 11)

    # round all the values in the list
    contactpoint1 = round_list(contactpoint1)
    endpoint1 = round_list(endpoint1)
    center1 = round_list(center1)
    contactpoint2 = round_list(contactpoint2)
    endpoint2 = round_list(endpoint2)
    center2 = round_list(center2)
    return contactpoint1,endpoint1,center1,contactpoint2,endpoint2,center2

# Example input data (uncomment for testing)
# path_to_forcedata = "C:\\Users\\16199\\Documents\\GitHub\\Prepare2Play-Learning-Experience-3\\data\\bcp_lr_CC_for03_Raw_Data.xlsx"
# data = pd.read_excel(path_to_forcedata)
# example_input = [[457, 643], [978, 648]]
# example_forcedata = data.iloc[18]
# find_contact_top(locationin=example_input, forcedata=example_forcedata)


