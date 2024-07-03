"""
Script: displayskeleton
    Add visual representation of segments and center of mass of body and segments.

Modules
    addskeleton: Add visual representation of segments and center of mass of body and segments.

Author:
    Casey Wiens
    cwiens32@gmail.com
"""


def addskeleton(file_vid, data, data_cm, segments,
                file_vid_n='skeletonvideo.mp4', samp_vid=240, flipy='yes', imout=False):
    """
    Function::: addskeleton
    	Description: Add visual representation of segments and center of mass of body and segments.
    	Details: imout must be True!!
            location: 'SkeletonOL' folder within location of file_vid

    Inputs
            file_vid: STR full file name of video
        data: DATAFRAME digitized data for each segment
            Column 0: frame
            Column 1+: digitized locations with x then y
        data_cm: DATAFRAME body and segment CM locations (prefe)
            Column 0: frame
            Column 1+: center of mass locations with x then y
        segments: DATAFRAME segment parameters obtained from segdim_deleva.py
        file_vid_n: STR full file name of new video (default: skeletonvideo.mp4)
        samp_vid: INT sampling rate of video (Hz) (default: 240)
        flipy: STR flip y-axis values to match (0,0) in upper left of video (default: 'no')
        imout: TRUE/FALSE option to output each frame as individual image (default: False)

    Outputs
        output1: image of each digitized frame with body cm, segment cm, and segment visually represented

    Dependencies
        cv2 (opencv)
        pandas
        numpy
        os
    """

    # Dependencies
    import cv2
    import pandas as pd
    import numpy as np
    import os

    #%% set up location to store images
    if imout is True:
        # if just file name was given
        if os.path.dirname(file_vid) == '':
            savefolder = 'SkeletonOL'
        else:
            savefolder = os.path.join(os.path.dirname(file_vid), 'SkeletonOL')
        # if folder does not exist
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        
    
    #%% load video file and initialize new video
    cap = cv2.VideoCapture(file_vid)
    # default resolutions of the frame are obtained.The default resolutions are system dependent.
    # we convert the resolutions from float to integer.
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    # define the codec and create VideoWriter object
    vid_out = cv2.VideoWriter(file_vid_n, cv2.VideoWriter_fourcc('M','P','4','V'),
                              samp_vid/4, (frame_width,frame_height))
    
    
    #%% flip y axis
    if flipy == 'yes':
        # digitized data
        data.loc[:, data.columns.str.contains('_y')] = frame_height - data.loc[:, data.columns.str.contains('_y')]
        # center of mass data
        data_cm.loc[:, data_cm.columns.str.contains('_y')] = frame_height - data_cm.loc[:, data_cm.columns.str.contains('_y')]
    
    #%% create video
    # find all frame numbers
    frame_num_list = data['frame']
    # create frame counter
    frame_cnt = 0
    # create frame list counter
    frame_list_cnt = 0
    
    while(True):
        ret, frame = cap.read()
        if ret == True:
            
            #%% apply skeleton on each image
            # if current frame is in data
            if any(frame_num_list.isin([frame_cnt])):
                
                # find what index current frame is
                frameloc = frame_num_list.isin([frame_cnt]).idxmax()
                
                #%% loop through digitized points
                for cnts in range(len(segments)):
                    # if segment doesn't contain nans
                    if (any(np.isnan(data.filter(regex = segments['origin'][cnts]).loc[frameloc,:])) or
                        any(np.isnan(data.filter(regex = segments['other'][cnts]).loc[frameloc,:]))):
                        pass
                    else:
                        # if it is head
                        if (segments.iloc[cnts,:]).name == 'head':
                            # origin
                            orig_loc = tuple(data.filter(regex = segments['origin'][cnts]).loc[frameloc,:].astype(int))
                            # other
                            oth_loc = tuple(data.filter(regex = segments['other'][cnts]).loc[frameloc,:].astype(int))
                            # draw segment with red line
                            frame = cv2.line(frame, orig_loc, oth_loc, (0,0,255), thickness=3)
                            # segment center of mass
                            segcm = data_cm.filter(regex = (segments.iloc[cnts,:]).name)
                            # display segment center of mass location
                            segcm_loc = tuple(segcm.loc[frameloc,:].astype(int))
                            frame = cv2.circle(frame, segcm_loc, 3, (0,0,0), -1)
                            
                        # if it is trunk
                        elif (segments.iloc[cnts,:]).name == 'trunk':
                            # origin
                            orig_loc = tuple(data.filter(regex = segments['origin'][cnts]).loc[frameloc,:].astype(int))
                            # other
                            oth = data.filter(regex = segments['other'][cnts])
                            # if both hips were located, use average
                            if len(oth.columns) > 2:
                                oth = pd.DataFrame({'x': oth.filter(regex='x').mean(axis = 1),
                                                    'y': oth.filter(regex='y').mean(axis = 1)})
                            # create tuple for other
                            oth_loc = tuple(oth.loc[frameloc,:].astype(int))
                            # draw segment with red line
                            frame = cv2.line(frame, orig_loc, oth_loc, (0,0,255), thickness=3)
                            # segment center of mass
                            segcm = data_cm.filter(regex = (segments.iloc[cnts,:]).name)
                            # display segment center of mass location
                            segcm_loc = tuple(segcm.loc[frameloc,:].astype(int))
                            frame = cv2.circle(frame, segcm_loc, 3, (0,0,0), -1)
                    
                        # if another segment
                        else:
                            # origin
                            orig = data.filter(regex = segments['origin'][cnts])
                            # segment center of mass
                            segcm = data_cm.filter(regex = (segments.iloc[cnts,:]).name)
                            # find if location exists in digitized data set
                            if len(orig.columns) > 0:
                                # find if left and right segments were specified
                                orig_l = orig.filter(regex = 'left')
                                orig_r = orig.filter(regex = 'right')
                                # find left and right segments
                                segcm_l = tuple(segcm.filter(regex = 'left').loc[frameloc,:].astype(int))
                                segcm_r = tuple(segcm.filter(regex = 'right').loc[frameloc,:].astype(int))
                            # frameloc
                            oth = data.filter(regex = segments['other'][cnts])
                            # find if location exists in digitized data set
                            if len(oth.columns) > 0:
                                # find if left and right segments were specified
                                oth_l = oth.filter(regex = 'left')
                                oth_r = oth.filter(regex = 'right')
                                
                            # if both origin and other locations exist
                            if (len(orig.columns)>0 and len(oth.columns)>0):
                                # if left segment exists
                                if (len(orig_l.columns)>0 or len(oth_l.columns)>0):
                                    # create tuple
                                    orig_loc = tuple(orig_l.loc[frameloc,:].astype(int))
                                    oth_loc = tuple(oth_l.loc[frameloc,:].astype(int))
                                    # draw segment with red line
                                    frame = cv2.line(frame, orig_loc, oth_loc, (0,0,255), thickness=3)
                                    # display segment center of mass location
                                    frame = cv2.circle(frame, segcm_l, 3, (0,0,0), -1)
                                # if right segment exists
                                if (len(orig_r.columns)>0 or len(oth_r.columns)>0):
                                    # create tuple
                                    orig_loc = tuple(orig_r.loc[frameloc,:].astype(int))
                                    oth_loc = tuple(oth_r.loc[frameloc,:].astype(int))
                                    # draw segment with red line
                                    frame = cv2.line(frame, orig_loc, oth_loc, (0,0,255), thickness=3)
                                    # display segment center of mass location
                                    frame = cv2.circle(frame, segcm_r, 3, (0,0,0), -1)
                                # if neither left or right exists
                                if (len(orig_l.columns)==0) and (len(orig_r.columns)==0):
                                    # create tuple
                                    orig_loc = tuple(orig.loc[frameloc,:].astype(int))
                                    oth_loc = tuple(orig.loc[frameloc,:].astype(int))
                                    # draw segment with red line
                                    frame = cv2.line(frame, orig_loc, oth_loc, (0,0,255), thickness=3)
                                    # display segment center of mass location
                                    segcm_loc = tuple(segcm.loc[frameloc,:].astype(int))
                                    frame = cv2.circle(frame, segcm_loc, 3, (0,0,0), -1)
                        
                        
                #%% display center of mass
                # if center of mass is not nan (from missing segment)
                if not any(np.isnan(data_cm[['body_x','body_y']].loc[frameloc])):
                    # create tuple
                    bodycm_loc = tuple(data_cm.filter(regex = 'body').loc[frameloc,:].astype(int))
                    # display center of mass location
                    frame = cv2.circle(frame, bodycm_loc, 8, (0,255,255), -1)
                
                """ iterate counter """
                frame_list_cnt += 1
            
            
            #%% save frame and add to video
            if imout is True:
                # create frame name
                framename = os.path.join(savefolder,
                                         os.path.basename(file_vid)[ :-4] + '_' + str(frame_cnt) + '.png')
                cv2.imwrite(framename, frame)

            # write the frame into the file
            vid_out.write(frame)
            # iterate frame number
            frame_cnt += 1
        else:
            break
            
    #%% when everything done, release the video capture and video write objects
    cap.release()
    vid_out.release()
    # closes all the frames
    cv2.destroyAllWindows()
