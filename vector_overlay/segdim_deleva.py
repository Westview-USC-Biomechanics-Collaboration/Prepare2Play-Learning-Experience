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

    # Dependencies
    import pandas as pd
    import numpy as np
    
    # create segment dimension dict
    segloc = pd.DataFrame(index=['head', 'trunk', 'upperarm', 'forearm',
                                 'hand', 'thigh', 'shank', 'foot'],
                          columns=['origin', 'other', 'joint_p', 'joint_d'],
                          dtype=str).reindex(columns=['origin', 'other',
                                             'joint_p', 'joint_d'])
    
    # create segment dimension dict
    segdim = pd.DataFrame(index=['head', 'trunk', 'upperarm', 'forearm',
                                 'hand', 'thigh', 'shank', 'foot'],
                          columns=['cmpos', 'massper', 'r_gyr'],
                          dtype=np.int64)
    
    #%% store location of origin and other for each segment
    # origin
    segloc['origin']['head'] = 'vertex'
    segloc['origin']['trunk'] = 'c7'
    segloc['origin']['upperarm'] = 'shoulder'
    segloc['origin']['forearm'] = 'elbow'
    segloc['origin']['hand'] = 'wrist'
    segloc['origin']['thigh'] = 'hip'
    segloc['origin']['shank'] = 'knee'
    segloc['origin']['foot'] = 'heel'
    # other
    segloc['other']['head'] = 'c7'
    segloc['other']['trunk'] = 'hip'
    segloc['other']['upperarm'] = 'elbow'
    segloc['other']['forearm'] = 'wrist'
    segloc['other']['hand'] = 'finger'
    segloc['other']['thigh'] = 'knee'
    segloc['other']['shank'] = 'ankle'
    segloc['other']['foot'] = 'toe'
    
    #%% store location of proximal joint and distal joint landmarks
    # origin
    segloc['joint_p']['head'] = None
    segloc['joint_p']['trunk'] = 'c7'
    segloc['joint_p']['upperarm'] = 'shoulder'
    segloc['joint_p']['forearm'] = 'elbow'
    segloc['joint_p']['hand'] = 'wrist'
    segloc['joint_p']['thigh'] = 'hip'
    segloc['joint_p']['shank'] = 'knee'
    segloc['joint_p']['foot'] = 'ankle'
    # other
    segloc['joint_d']['head'] = 'c7'
    segloc['joint_d']['trunk'] = 'hip'
    segloc['joint_d']['upperarm'] = 'elbow'
    segloc['joint_d']['forearm'] = 'wrist'
    segloc['joint_d']['hand'] = None
    segloc['joint_d']['thigh'] = 'knee'
    segloc['joint_d']['shank'] = 'ankle'
    segloc['joint_d']['foot'] = None

    #%% location of center of mass length percentages for each segment
    # if female..
    if sex == 'f':
        segdim['cmpos']['head'] = 0.4841
        segdim['cmpos']['trunk'] = 0.4964
        segdim['cmpos']['upperarm'] = 0.5754
        segdim['cmpos']['forearm'] = 0.4592
        segdim['cmpos']['hand'] = 0.3502
        segdim['cmpos']['thigh'] = 0.3612
        segdim['cmpos']['shank'] = 0.4416
        segdim['cmpos']['foot'] = 0.4014
    
    else:
        segdim['cmpos']['head'] = 0.5002
        segdim['cmpos']['trunk'] = 0.5138
        segdim['cmpos']['upperarm'] = 0.5772
        segdim['cmpos']['forearm'] = 0.4608
        segdim['cmpos']['hand'] = 0.3691
        segdim['cmpos']['thigh'] = 0.4095
        segdim['cmpos']['shank'] = 0.4459
        segdim['cmpos']['foot'] = 0.4415
        
    #%% mass percents for each segment
    # if female..
    if sex == 'f':
        segdim['massper']['head'] = 0.0668
        segdim['massper']['trunk'] = 0.4257
        segdim['massper']['upperarm'] = 0.0255
        segdim['massper']['forearm'] = 0.0138
        segdim['massper']['hand'] = 0.0056
        segdim['massper']['thigh'] = 0.1478
        segdim['massper']['shank'] = 0.0481
        segdim['massper']['foot'] = 0.0129
    
    else:
        segdim['massper']['head'] = 0.0694
        segdim['massper']['trunk'] = 0.4346
        segdim['massper']['upperarm'] = 0.0271
        segdim['massper']['forearm'] = 0.0162
        segdim['massper']['hand'] = 0.0061
        segdim['massper']['thigh'] = 0.1416
        segdim['massper']['shank'] = 0.0433
        segdim['massper']['foot'] = 0.0137
        
    #%% radius of gyration for each segment
    # if female..
    if sex == 'f':
        segdim['r_gyr']['head'] = 0.271
        segdim['r_gyr']['trunk'] = 0.307
        segdim['r_gyr']['upperarm'] = 0.278
        segdim['r_gyr']['forearm'] = 0.263
        segdim['r_gyr']['hand'] = 0.241
        segdim['r_gyr']['thigh'] = 0.369
        segdim['r_gyr']['shank'] = 0.271
        segdim['r_gyr']['foot'] = 0.299
    
    else:
        segdim['r_gyr']['head'] = 0.303
        segdim['r_gyr']['trunk'] = 0.328
        segdim['r_gyr']['upperarm'] = 0.285
        segdim['r_gyr']['forearm'] = 0.278
        segdim['r_gyr']['hand'] = 0.285
        segdim['r_gyr']['thigh'] = 0.329
        segdim['r_gyr']['shank'] = 0.255
        segdim['r_gyr']['foot'] = 0.257
        
    #%% join data tables
    segments = segloc.join(segdim)

    #%% return data frame
    return segments