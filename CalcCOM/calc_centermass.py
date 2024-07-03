#TODO ask casey about the inputs and outputs in the code**
"""
Script: calc_centermass
    Calculate center of mass for each segment and whole body/system.

Modules
    main: Run functions to caclulate CM
    calc_indsegcm: Calculate center of mass for an individual segment
    calc_segcm: Calculate the center of mass of all the segments
    calc_cm: Calculate overall body center of mass

Author:
    Casey Wiens
    cwiens32@gmail.com
"""


def calc_indsegcm(cmpos, origin_loc, other_loc):
    """
    Function::: calc_indsegcm
    	Description: Calculate center of mass for an individual segement
    	Details:

    Inputs
        cmpos:
        origin_loc:
        other_loc:

    Outputs
        segcm:

    Dependencies
        None
    """
    # Dependencies

    # convert column names
    origin_loc.columns = ['x', 'y']
    other_loc.columns = ['x', 'y']
    # calculate center of mass location in x and y
    # cm_p% * (cm_d - cm_p) + cm_p
    segcm = cmpos * (other_loc - origin_loc) + origin_loc

    return segcm
    

def calc_segcm(datain, segments):
    """
    Function::: calc_segcm
        Description: Calculate center of mass of an individual segement
        Details:

    Inputs
        datain: DATAFRAME Digitized data with joint centers
        segments: DATAFRAME segment parameters obtained from segdim_deleva.py (default=None)

    Outputs
        dataout: DATAFRAME with each segements center of mass calculated

    Dependencies
        pandas
    """

    # Dependencies
    import pandas as pd

    # initialize data out
    dataout = pd.DataFrame(datain.iloc[:,0])
    
    # loop through segments
    for cnt in range(len(segments)):
        # if it is head
        if (segments.iloc[cnt,:]).name == 'head':
            # origin
            orig = datain.filter(regex = segments['origin'][cnt])
            # other
            oth = datain.filter(regex = segments['other'][cnt])
            # calculate center of mass location of segment
            seg = calc_indsegcm(segments['cmpos'][cnt], orig, oth)
            # rename columns
            seg.columns = [(segments.iloc[cnt,:]).name + '_x',
                             (segments.iloc[cnt,:]).name + '_y']
            # add column to data out
            dataout = dataout.join(seg)
            
        # if it is trunk
        elif (segments.iloc[cnt,:]).name == 'trunk':
            # origin
            orig = datain.filter(regex = segments['origin'][cnt])
            # other
            oth = datain.filter(regex = segments['other'][cnt])
            # if both hips were located, use average
            if len(oth.columns) > 2:
                oth = pd.DataFrame({'x': oth.filter(regex='x').mean(axis = 1),
                                    'y': oth.filter(regex='y').mean(axis = 1)})
            # calculate center of mass location of segment
            seg = calc_indsegcm(segments['cmpos'][cnt], orig, oth)
            # rename columns
            seg.columns = [(segments.iloc[cnt,:]).name + '_x',
                             (segments.iloc[cnt,:]).name + '_y']
            # add column to data out
            dataout = dataout.join(seg)
            
        # if another segment
        else:
            # origin
            orig = datain.filter(regex = segments['origin'][cnt])
            # find if location exists in digitized data set
            if len(orig.columns) > 0:
                # find if left and right segments were specified
                orig_l = orig.filter(regex = 'left')
                orig_r = orig.filter(regex = 'right')
            # other
            oth = datain.filter(regex = segments['other'][cnt])
            # find if location exists in digitized data set
            if len(oth.columns) > 0:
                # find if left and right segments were specified
                oth_l = oth.filter(regex = 'left')
                oth_r = oth.filter(regex = 'right')
              
            # if both origin and other locations exist
            if (len(orig.columns)>0 and len(oth.columns)>0):
                # if left segment exists
                if (len(orig_l.columns)>0 or len(oth_l.columns)>0):
                    # calculate center of mass location of segment
                    seg_l = calc_indsegcm(segments['cmpos'][cnt], orig_l, oth_l)
                    # rename columns
                    seg_l.columns = [(segments.iloc[cnt,:]).name + '_left_x',
                                     (segments.iloc[cnt,:]).name + '_left_y']
                    # add column to data out
                    dataout = dataout.join(seg_l)
                # if right segment exists
                if (len(orig_r.columns)>0 or len(oth_r.columns)>0):
                    # calculate center of mass location of segment
                    seg_r = calc_indsegcm(segments['cmpos'][cnt], orig_r, oth_r)
                    # rename columns
                    seg_r.columns = [(segments.iloc[cnt,:]).name + '_right_x',
                                     (segments.iloc[cnt,:]).name + '_right_y']
                    # add column to data out
                    dataout = dataout.join(seg_r)
                # if neither left or right exists
                if (len(orig_l.columns)==0) and (len(orig_r.columns)==0):
                    # calculate center of mass location of segment
                    seg = calc_indsegcm(segments['cmpos'][cnt], orig, oth)
                    # rename columns
                    seg.columns = [(segments.iloc[cnt,:]).name + '_x',
                                     (segments.iloc[cnt,:]).name + '_y']
                    # add column to data out
                    dataout = dataout.join(seg)

    return dataout


def calc_cm(segcm, segments):
    """
        Function::: calc_cm
        	Description: Calculate the center of mass for all segments
        	Details: Circles through Deleva parameters

        Inputs
            segcm: DATAFRAME Contains CM of each segment
            segments: DATAFRAME segment parameters obtained from segdim_deleva.py (default=None)

        Outputs
            bodycm: DATAFRAME location of body/system center of mass

        Dependencies
            pandas
        """

    # Dependencies
    import pandas as pd

    # initialize total body and total mass variable
    totalbody = pd.DataFrame(segcm.iloc[:,0])
    totalmass = 0
    
    # loop through segments
    for cnt in range(len(segments)):
        # find columns of current segment
        seg = segcm.filter(regex = segments.iloc[cnt,:].name)
        # multiply segment center of mass location by % mass segment of total body
        seg = seg * segments['massper'][cnt]
        # add current segment to total body data frame
        totalbody = totalbody.join(seg)
        # find total % segment mass to divide by
        segmass = int(len(seg.columns)/2) * segments['massper'][cnt]
        # add current segment mass to total
        totalmass += segmass
    
    # calculate center of mass of body
    # sum(mi * xcmi) / M
    x = pd.DataFrame({'body_x': totalbody.filter(regex = '_x').sum(axis=1, skipna=False) / totalmass})
    y = pd.DataFrame({'body_y': totalbody.filter(regex = '_y').sum(axis=1, skipna=False) / totalmass})
    # join datatables
    bodycm = pd.DataFrame(segcm).join(x).join(y)
    
    return bodycm



def main(data, segments=None, sex=None):
    """
    Function::: main
    	Description: Calculate center of mass a participant
    	Details: USER NEEDS TO PROVIDE EITHER SEGMENTS OR SEX

    Inputs
        data: DATAFRAME digitized data for each segment
            Column 0: frame
            Column 1+: digitized locations with x then y
        segments: DATAFRAME segment parameters obtained from segdim_deleva.py (default=None)
        sex: STR input to gather segment parameters [m (male) or f (female)] (default=None)

    Outputs
        cm_body: DATAFRAME location of body/system center of mass

    Dependencies
        CalcCOM
    """

    # Dependencies
    from CalcCOM.segdim_deleva import segmentdim

    # gather segment dimension parameters if not provided
    if segments is None:
        # find segment parameters
        segments = segmentdim(sex)
    # calculate segments' center of mass
    cm_seg = calc_segcm(data, segments)
    # calculate body center of mass
    cm_body = calc_cm(cm_seg, segments)

    return cm_body

