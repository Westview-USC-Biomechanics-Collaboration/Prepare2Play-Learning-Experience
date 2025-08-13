"""
Script: segdim_deleva
    Description of the overall purpose of the script.

Modules
    segmentdim: Create data frame containing segment center of mass length and percent weight based on de Leva 1996.

Author:
    Casey Wiens
    cwiens32@gmail.com
"""

def segmentdim(sex):
    """
    Function::: segmentdim
    	Description: Create data frame containing segment center of mass length and percent weight based on de Leva 1996.
    	Details: de Leva. Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters. 1996
            Based on digitized points:
                segment:    origin                  -   other
                head:       vertex                  -   cervicale (C7)
                trunk:      cervical (C7)           -   mid-hip
                upperarm:   shoulder joint center   -   elbow joint center
                forearm:    elbow joint center      -   stylion (distal point radial styloid)
                hand:       sylion (see forearm)    -   3rd dactylion (tip of 3rd digit)
                thigh:      hip joint center        -   knee joint center
                shank:      knee joint center       -   lateral malleolus
                foot:       heel                    -   acropodion (tip of longest toe - 1st or 2nd)

    Inputs
        sex: STR segmental parameters for which gender ('f' or 'm')

    Outputs
        segments: DATAFRAME contains origin and other location for segment definition,
            proximal and distal joints (primarily for digitized point),
            as well as segmental center of mass position (cmpos),
            percent weight (massper), and sagittal radii of gyration (r_gyr).

    Dependencies
        pandas
        numpy
    """

    import pandas as pd
    import numpy as np

    # Define segments
    segments_list = ['head', 'trunk', 'upperarm', 'forearm', 'hand', 'thigh', 'shank', 'foot']

    # Create segment location and dimension dataframes
    segloc = pd.DataFrame(index=segments_list, columns=['origin', 'other', 'joint_p', 'joint_d'], dtype=object)
    segdim = pd.DataFrame(index=segments_list, columns=['cmpos', 'massper', 'r_gyr'], dtype=np.float64)

    # Set segment location info
    segloc.loc['head', ['origin', 'other']]     = ['vertex', 'c7']
    segloc.loc['trunk', ['origin', 'other']]    = ['c7', 'hip']
    segloc.loc['upperarm', ['origin', 'other']] = ['shoulder', 'elbow']
    segloc.loc['forearm', ['origin', 'other']]  = ['elbow', 'wrist']
    segloc.loc['hand', ['origin', 'other']]     = ['wrist', 'finger']
    segloc.loc['thigh', ['origin', 'other']]    = ['hip', 'knee']
    segloc.loc['shank', ['origin', 'other']]    = ['knee', 'ankle']
    segloc.loc['foot', ['origin', 'other']]     = ['heel', 'toe']

    segloc.loc['head', ['joint_p', 'joint_d']]      = [None, 'c7']
    segloc.loc['trunk', ['joint_p', 'joint_d']]     = ['c7', 'hip']
    segloc.loc['upperarm', ['joint_p', 'joint_d']]  = ['shoulder', 'elbow']
    segloc.loc['forearm', ['joint_p', 'joint_d']]   = ['elbow', 'wrist']
    segloc.loc['hand', ['joint_p', 'joint_d']]      = ['wrist', None]
    segloc.loc['thigh', ['joint_p', 'joint_d']]     = ['hip', 'knee']
    segloc.loc['shank', ['joint_p', 'joint_d']]     = ['knee', 'ankle']
    segloc.loc['foot', ['joint_p', 'joint_d']]      = ['ankle', None]

    # Segment parameters based on sex
    if sex == 'f':
        cmpos = [0.4841, 0.4964, 0.5754, 0.4592, 0.3502, 0.3612, 0.4416, 0.4014]
        massper = [0.0668, 0.4257, 0.0255, 0.0138, 0.0056, 0.1478, 0.0481, 0.0129]
        r_gyr = [0.271, 0.307, 0.278, 0.263, 0.241, 0.369, 0.271, 0.299]
    else:
        cmpos = [0.5002, 0.5138, 0.5772, 0.4608, 0.3691, 0.4095, 0.4459, 0.4415]
        massper = [0.0694, 0.4346, 0.0271, 0.0162, 0.0061, 0.1416, 0.0433, 0.0137]
        r_gyr = [0.303, 0.328, 0.285, 0.278, 0.285, 0.329, 0.255, 0.257]

    segdim.loc[:, 'cmpos'] = cmpos
    segdim.loc[:, 'massper'] = massper
    segdim.loc[:, 'r_gyr'] = r_gyr

    # Join data tables
    segments = segloc.join(segdim)

    return segments
