import cv2
import numpy as np
import pandas as pd
import time
import os
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import GUI.models.video_state
plt.ioff()
def plate_transformation_matrix():
    # # Load video
    # cap = cv2.VideoCapture(c.path_video)
    
    # #------ Determine appropriate threshold for making mask to find plates -------#
    # print('\n-------------- Finding Threshold for Finding Plates ----------------')
    # threshold_list = []
    # for frame_number in range(c.num_frames_for_threshold):
    #     ##### Get current frame and extract crop of Red Channel
    #     print('Frame Number: ', str(frame_number))
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    #     ret, frame = cap.read()
        
    #     cropRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[c.crop_y0:c.crop_y1, c.crop_x0:c.crop_x1, :]
    #     cropR = cropRGB[:, :, 0]
        
    #     ##### Determine threshold to use for making mask from Red Channel crop
    #     # Sweep through possible intensity thresholds
    #     # Use each as a threshold and determine number of positive pixels
    #     # Note, thresholding 0/255 instead of 0/1 to make visualization easier
    #     intensity_list = []
    #     positive_count_list = []
    #     for intensity in range(0,256):
    #         ret, maskR = cv2.threshold(cropR, intensity, 255, cv2.THRESH_BINARY)
    #         intensity_list.append(intensity)
    #         positive_count_list.append(maskR.sum()/255)
        
    #     # Calculate profile of positive counts as a function of intensity
    #     # Calculate 1st and 2nd derivatives as delta of counts (intensity step = 1)
    #     # Find peaks associated with lower and upper thresholds around the
    #     # the intensities corrsponding to the plate perimeter.
    #     #   As the threshold increases there are three main zones:
    #     #       1) There is a large drop associated with the vast majority of pixels
    #     #           below threshold (i.e., black platform and green background)
    #     #       2) A range of thresholds where the counts drops gradually because
    #     #           the majority of the counts are due to the plate perimeter so 
    #     #           the mask is not changing much
    #     #       3) A range where the counts drop a bit faster now that the
    #     #           threshold is high enough to remove portions of the perimeter
    #     #   By smoothing the absolute value of the 2nd derivative, we create a
    #     #   profile where 2 or 3 peaks are detectable:
    #     #       1) Large peak associated with zone 1 above.
    #     #           We want the threshold toabove this peak.
    #     #       2) A small wide peak associated with zone 2 may or may not be found.
    #     #           We don't need to find a peak in this zone.
    #     #       3) A medium peak associated with zone 3.
    #     #           We want the threshold to be below this peak.
    #     #   We select the threshold by averaging the intensities from two limits:
    #     #       1) Lower limit based on the intensity associated with the
    #     #           right side of the zone 1 peak (highest intensity for zone 1)
    #     #       2) Upper limit based on the intensity associated with the
    #     #           left side of the zone 3 peak (lowest intensity for zone 3)
    #     #
    #     i_data = [0]
    #     delta1_data=[0]
    #     delta2_data=[0]
    #     abs_delta2_data = [0]
    #     # Manually tabulate deltas
    #     for i in range(1,256):
    #         i_data.append(i)
    #         delta1 = positive_count_list[i] - positive_count_list[i-1]
    #         delta1_data.append(delta1)
    #         delta2 = delta1_data[i] - delta1_data[i-1]
    #         delta2_data.append(delta2)
    #         abs_delta2_data.append(abs(delta2))
            
    #     # Smooth the curve to make it easier to find just 2 or 3 peaks
    #     sg = signal.savgol_filter(abs_delta2_data, 30, 1)
        
    #     # Only going to use cases where we only find 2 or 3 peaks
    #     # Assume the first peak is for zone 1 and last peak is for zone 3
    #     # Doesn't matter if a peak is found for zone 2
    #     peaks = signal.find_peaks(sg, width=10)
    #     num_peaks = len(peaks[0])
    #     threshold = 0
    #     if num_peaks ==2 or num_peaks==3:
    #         print('Number of peaks: ' + str(num_peaks))
    #         if c.make_plots==True: plt.plot(i_data, sg)
    #         for p in range(len(peaks[0])):
    #             x_peak = peaks[0][p]
    #             print('Peak at ' + str(x_peak))
    #             if c.make_plots==True: plt.vlines(x=x_peak, ymin=plt.ylim()[0], ymax = plt.ylim()[1], colors='red', linestyles='--')
    
    #         left_limit = peaks[1]['right_ips'][0]
    #         right_limit = peaks[1]['left_ips'][-1]
    #         threshold = (left_limit + right_limit)//2 
    #         threshold_list.append(threshold)
    #         if c.make_plots == True:
    #              plt.vlines(x=threshold, ymin=plt.ylim()[0], ymax = plt.ylim()[1], colors='green', linestyles='solid')
    #              plt.title('Intensity Threshold for Frame: ' + str(frame_number))
    #              plt.show()
        
    #     if threshold > 0:
    #         print('Threshold for frame ' + str(frame_number) + ' : ' +str(threshold))
    #     else:
    #         print('No threshold found for frame ' + str(frame_number))
    
    # # Select a robust threshold based on median value
    # intensity_threshold = np.median(threshold_list)
    # print('\nThreshold  for plate finding: ' + str(intensity_threshold))
    # #-----------------------------------------------------------------------------#
    
    # Select threshold
# =============================================================================
#     if c.use_auto_threshold == True:
#         print('Use Auto Threshold')
#         intensity_threshold = get_auto_threshold()
#     else:
#         print('Use Fixed Threshold')
#         intensity_threshold = c.fixed_threshold
# 
#     print(intensity_threshold)
# =============================================================================
    #---------------- Find Corners of Two-Plate System ---------------------------#

    
    print('\n-----------------Finding Corners --------------------------------')
    # Load video
    cap = cv2.VideoCapture(c.path_video)
    
# =============================================================================
#     min_area = c.min_plate_area
#     print('################### Min Area: ' + str(min_area))
#     corners_list = []
#     for frame_number in range(c.num_frames_for_corners):
#         ##### Get current frame and extract crop of Red Channel
#         print('Frame Number: ', str(frame_number))
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
#         ret, frame = cap.read()
#         if c.top_view: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#         
#         cropRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[c.crop_y0:c.crop_y1, c.crop_x0:c.crop_x1, :]
#         cropR = cropRGB[:, :, 0]
#         
#         # Create mask for finding contours
#         # Use erode and dialte to improve robustness
#         ret, mask = cv2.threshold(cropR, intensity_threshold, 255, cv2.THRESH_BINARY)
#         mask = cv2.dilate(mask, kernel=np.ones((5, 5), np.uint8), iterations=1)
#         mask = cv2.erode(mask, kernel=np.ones((3, 3), np.uint8), iterations=1)
#         mask = cv2.dilate(mask, kernel=np.ones((5, 5), np.uint8), iterations=1)
#         mask = cv2.erode(mask, kernel=np.ones((3, 3), np.uint8), iterations=1)
#         #plt.imshow(mask, cmap='gray'), plt.show()
#         
#         # Find set of contours for the mask
#         contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
#         
#         # Process appropriate contours to find corners
#         # Require minimum area to avoid contours thaat are too small
#         for contour in contours:
#             area = cv2.contourArea(contour)
#             if area > min_area:
#                 print(area)
#                 #img = cropRGB.copy()
#                 #cv2.drawContours(img, [contour], -1, (0,255,0), 2)
#                 #plt.imshow(img), plt.show()
#                 
#                 contourFilled = np.zeros((cropR.shape[0], cropR.shape[1])).astype(np.uint8)
#                 cv2.drawContours(contourFilled, [contour], -1, 255, cv2.FILLED)
#                 #plt.imshow(contourFilled, cmap='gray'), plt.show()
#                 
#                 # Find the four corners via template matching
#                 res_11 = cv2.matchTemplate(contourFilled, c.template_11, cv2.TM_SQDIFF)
#                 min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_11)
#                 corner_11x = min_loc[0] + c.offset_11[0]
#                 corner_11y = min_loc[1] + c.offset_11[1]
#         
#                 res_12 = cv2.matchTemplate(contourFilled, c.template_12 ,cv2.TM_SQDIFF)
#                 min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_12)
#                 corner_12x = min_loc[0] + c.offset_12[0]
#                 corner_12y = min_loc[1] + c.offset_12[1]
#         
#                 res_21 = cv2.matchTemplate(contourFilled, c.template_21, cv2.TM_SQDIFF)
#                 min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_21)
#                 corner_21x = min_loc[0] + c.offset_21[0]
#                 corner_21y = min_loc[1] + c.offset_21[1]
#         
#                 res_22 = cv2.matchTemplate(contourFilled, c.template_22, cv2.TM_SQDIFF)
#                 min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_22)
#                 corner_22x = min_loc[0] + c.offset_22[0]
#                 corner_22y = min_loc[1] + c.offset_22[1]                   
#                 
#                 corners = np.array([[corner_11x, corner_11y], [corner_12x, corner_12y], [corner_22x, corner_22y], [corner_21x, corner_21y]])
#                 corners = corners.reshape(1, 4, 2)
#                 corners_list.append(corners)
#                 
#                 # Print plot showing location based on identified corners
#                 if c.make_plots == True:
#                     imgC = cropRGB.copy()
#                     cv2.polylines(imgC, corners, True, (0,255,0), 2)
#                     plt.imshow(imgC)
#                     plt.title('Plate Location for Frame: ' + str(frame_number)) 
#                     plt.show()
#                 
#             
#     # Report Results of Corner Detection
#     print('\nCorner locations found for each frame (by row):')
#     results = np.array(corners_list).reshape(len(corners_list), 4, 2)
#     for i in range(4):
#         for j in range(2):
#             print(results[:, i, j])
#     
#     print('\nSpans (Max - Min) for each corner:')
#     print(np.max(results, axis=0) - np.min(results, axis=0))
#     
#     corner_locations = np.median(results, axis=0)
#     print('\nCorner locaations in the context of the cropped image used to find plates:')
#     print(corner_locations)
#     
#     ### Adjust corner locations to correspond to the location in full frame images
#     # Adjustment based on the origin of the crop location
#     corner_locations += [c.crop_x0, c.crop_y0]
#     print('\nCorner locaations in the context of the full frame images:')
#     print(corner_locations)
# =============================================================================
    
    
    #------------------- Calculate Transformation Matrix -------------------------#
    # Transformation Matrix will map a pressure point location on either of the 
    # plates to the location in the full frame image.
    #
    # The pressure point locations for each plate are expressed as:
    #   1) A value from -0.450 to +0.450 meters for Ay
    #       Note, this is for the horizontal (X-direction) of the front view
    #   2) A value from -0.300 to +0.300 meters for Ax
    #       Note, this is for the vertical (Y-direction) of the front view
    # The center of each plate is expressed as the (0, 0) origin of the plate
    #
    # We are treating the two-side-by-side plates as a single unit for mapping
    # Plate 1 is on the left when looking at the front view video.
    # There is a small gap between plates to isolate them from each other.
    # We will assume this gap is 0.004 meters, but this value can be changed.
    # We will make the origin of the two-plate system to be in the center of this
    # gap between the plates.
    # Therefore, the pressure point locations need to:
    #   1) Span from -0.902 to +0.902 meters horizontaly (Ay)
    #       a) Need to subtract 0.452 from Plate1 Ay
    #       b) Need to add 0.452 to Plate2 Ay
    #   2) Span from -0.300 to +0.300 vertically (Ax)
    #       No adjustment is needed for Ax of either plate
    #
    # The transformation Matrix will map:
    #   From:   The ideal rectangle that is 1.804 X 0.600 meters in size
    #   To:     The quadrilateral shape seen in the image that is distorted due to
    #           the camera perspective; defined by the 4 corners we found.
    #
    # Ideal Rectangle described as +/- deltas from center
    delta_x = 0.902
    delta_y = 0.300
    
    pts_rectangle = np.float32([[-delta_x, -delta_y], [delta_x, -delta_y],
                                [-delta_x, delta_y], [delta_x, delta_y]])
    # Define yellow color range in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    # Open the video
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError(f"Could not read a frame from: {c.path_video}")

    # Load an image instead of a video
    # frame = cv2.imread("vector_overlay\IMG_2518.jpg")

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a binary mask where yellow colors are white
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("original", 800, 600)  # Set window size
    cv2.imshow("original", mask)


    # Optional: closing to seal any final small gaps
    # Horizontal kernel to connect horizontal lines
    kernel_h = np.ones((1, 200), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h)
    kernel_v = np.ones((1, 1), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_v)
    
    cv2.namedWindow("one", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("one", 800, 600)  # Set window size
    cv2.imshow("one", mask)
    

    h, w = mask.shape[:2]
    offset_x, offset_y = 0,0
# =============================================================================
#     if not c.top_view:
#         y1, y2 = int(0.6 * h), int(0.9 * h)
#         x1, x2 = int(0.25 * w), int(0.75 * w)
#         offset_x, offset_y = x1, y1
#         roi = mask[y1:y2, x1:x2]
#     else:
# =============================================================================
    roi = mask

    

    cv2.namedWindow("kernel observation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("kernel observation", 800, 600)  # Set window size
    cv2.imshow("kernel observation", roi)

    # Find all contours in the mask
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare to save coordinates
    coords = []

    # Minimum area to filter out noise
    min_area = 2000
    # max_area = 30000

    #save corners
    for contour in contours:
        if not c.top_view:
            contour += [offset_x, offset_y]
        hull = cv2.convexHull(contour)
        area = cv2.contourArea(hull)

        # Approximate contour to polygon to reduce points
        epsilon = 0.01 * cv2.arcLength(hull, True)  # adjust for precision
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if area > min_area:

            corners = approx.reshape(-1, 2)  # shape (num_points, 2)

            if len(corners) == 4:
                cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)  # Blue polygon outline
                for corner in corners:
                    x, y = corner
                    coords.append([x, y])

    coords_one = sorted(coords, key=lambda x: x[0])[0:2]
    coords_two = sorted(coords, key=lambda x: x[0])[2:]
    coords_one = sorted(coords_one, key=lambda x: x[1])
    coords_two = sorted(coords_two, key=lambda x: x[1])
    coords = coords_one + coords_two
    print(coords)

    print(f"Detected {len(coords)} corners.")
    if len(coords) < 4:
        print("Error: Not enough corners detected. Please try again.")

    # find remaining four points in the middle
    # coords.append([(coords[0][0] + coords[2][0])/2 - 10, (coords[0][1] + coords[2][1])/2])
    # coords.append([(coords[0][0] + coords[2][0])/2 + 10, (coords[0][1] + coords[2][1])/2])
    # coords.append([(coords[1][0] + coords[3][0])/2 - 10, (coords[1][1] + coords[3][1])/2])
    # coords.append([(coords[1][0] + coords[3][0])/2 + 10, (coords[1][1] + coords[3][1])/2])
    
    for out in coords:
        cv2.circle(frame, (int(out[0]), int(out[1])), 5, (0, 0, 255), -1)
    
    # Show the result
    cv2.namedWindow("Detected Yellow Rectangles", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detected Yellow Rectangles", 800, 600)  # Set window size
    cv2.imshow("Detected Yellow Rectangles", frame)

    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()

    
    # Use corners in image to describe perspective quadrilateral
    TL = coords[0]
    TR = coords[2]
    BR = coords[3]
    BL = coords[1]
    
    pts_perspective = np.float32([TL, TR, BL, BR])
    
    # Calculate transformation matrix (Rectangle --> perspective)
    matrix = cv2.getPerspectiveTransform(pts_rectangle, pts_perspective)
    print('\nTransformation Matrix:')
    print(matrix)
    
    # Done processing video, so release the resource
    cap.release()
    
    return matrix

def process_force_file(parent_path, force_file):    
    # Read data from txt file, need to avoid a header section
    # Column identifiers occupy two rows (name and units) which results in
    # all values being imported as strings.
    # Remove the row with units and convert data to numeric values
    # Also rename the columns to indicate the force plate used
    # experiment_name = video_file.removesuffix('.mov').removesuffix('.MOV').removesuffix('.MP4').removesuffix('.mp4')
    # save_force_data = True
    # file_force_data_dataframe = experiment_name + '_{Trimmed_Data}.csv'
    # path_force_data_dataframe = os.path.join(parent_path, output_folder, file_force_data_dataframe)

    trim_force_data = True
    dead_time = 0.25
    significant_force = 50

    path_force = os.path.join(parent_path, force_file)

    # Specify the column names for the force data
    # Assumes data is collected for three force plates
    # Third force plate is actually for the alignment signal, but gets captured
    # in the format of force plate data
    force_dict = {'abs time (s)': 'Time(s)',
                'Fx': 'FP1_Fx', 'Fy': 'FP1_Fy', 'Fz': 'FP1_Fz', '|Ft|': 'FP1_|F|', 'Ax': 'FP1_Ax', 'Ay': 'FP1_Ay',
                'Fx.1': 'FP2_Fx', 'Fy.1': 'FP2_Fy', 'Fz.1': 'FP2_Fz', '|Ft|.1': 'FP2_|F|', 'Ax.1': 'FP2_Ax', 'Ay.1': 'FP2_Ay',
                'Fx.2': 'FP3_Fx', 'Fy.2': 'FP3_Fy', 'Fz.2': 'FP3_Fz', '|Ft|.2': 'FP3_|F|', 'Ax.2': 'FP3_Ax', 'Ay.2': 'FP3_Ay'}

    df_force = pd.read_csv(path_force, header=17, delimiter='\t')
    df_force = df_force.drop(0)
    df_force = df_force.astype(float)
    df_force.rename(columns=force_dict, inplace=True)
    
    # Create a clean alignment signal that toggles between +/- 1
    # ON signals have large positive values and OFF signals have large
    # magnitude negative values, so can threshold with sign function
    df_force['FP_LED_Signal'] = np.sign(df_force['FP3_Fz'])
    
    # Map the pressure locations from the plates to the video
    # Pressure locations given as Ax & Ay for each plate, where values are in meters
    # For front view video:
    #   Ax ranges from -0.300 to +0.300 and corresponds to the the direction away or
    #   toward the camera, and corresponds to the vertical location in the video.
    #
    #   Ay ranges from -0.450 to +0.450 and corresponds to the horizontal location
    #   in the video.  Plate 1 is on the left as seen in the front view video.
    #
    #   The two plates side-by-side are treated as a single wider plate for mapping.
    #   Assume gap of 0.004 m between plates and consider the (0,0) origin in the center
    #
    #       1) X-direction (horizontal) in video uses Ay values
    #           a) Subtract 0.452 from the Plate 1 values
    #           b) Add 0.452 to the Plate 2 values
    #       2) Y-direction (vertical) in video uses Ax values
    #           a) No modification needed to Ax for either plate
    #
    df_force['FP1_X_m'] = df_force['FP1_Ay'] - 0.452
    df_force['FP1_Y_m'] = df_force['FP1_Ax']
    
    df_force['FP2_X_m'] = df_force['FP2_Ay'] + 0.452
    df_force['FP2_Y_m'] = df_force['FP2_Ax']
    
    # Use the transformation matrix to map the pressure points from the plates to
    # the pixel locations in the full frame video images.
    FP1_PP_list = []
    FP2_PP_list = []
    for index, row in df_force.iterrows():
        x1 = row['FP1_X_m']
        y1 = row['FP1_Y_m']
        rect1_pt = np.array([[[x1, y1]]]).astype(float)
        trans1_pt = cv2.perspectiveTransform(rect1_pt, M).round().astype(np.int64)
        FP1_PP_list.append(trans1_pt)
        
        x2 = row['FP2_X_m']
        y2 = row['FP2_Y_m']
        rect2_pt = np.array([[[x2, y2]]]).astype(float)
        trans2_pt = cv2.perspectiveTransform(rect2_pt, M).round().astype(np.int64)
        FP2_PP_list.append(trans2_pt)
    
    df_force['FP1_PP'] = FP1_PP_list
    df_force['FP2_PP'] = FP2_PP_list

    
    
    # Trimming force data stream
    # The force data can have relatively long periods before and/or after the
    # subject's activities.  To minimize the amount of "dead time" before and after
    # the activities of interest, a flag can be set in the config file to indicate
    # that the force data should be trimmed to:
    #   1) Start a specified time before the first significant Fz force on either plate
    #   2) Stop after the last significant F_total force on either plate
    # The threshold for significant F_total is also specified in the config file
    
    # Calculate max force for either plate
    df_force['MaxForce'] = df_force[['FP1_|F|', 'FP2_|F|']].max(axis=1)
    
    if trim_force_data == True:
        print('\n----- Trimming Force Data -----')
        print('Significant Force Threshold: ' + str(significant_force))
        print('Amount of dead time to keep: ' + str(dead_time))
    
              
        # Find the time points for trimming and trim the data set
        first = df_force[df_force['MaxForce'] >= significant_force].iloc[0]['Time(s)'] - dead_time
        last = df_force[df_force['MaxForce'] >= significant_force].iloc[-1]['Time(s)'] + dead_time
        
        force_start = max( first, df_force['Time(s)'].iloc[0])
        force_stop  = min( last, df_force['Time(s)'].iloc[-1])
        
        df_trimmed = df_force.loc[(df_force['Time(s)'] >= force_start) & (df_force['Time(s)'] <= force_stop )]
      
    else:
        print('\n----- Not Trimming Force Data -----')
        df_trimmed = df_force.copy()
    
    # Save files, of desired
    # if save_force_data == True:
    #     print('Saving force data information')
    #     df_trimmed.to_csv(path_force_data_dataframe)
    
    
    return df_trimmed

def find_led_location(self, path_video, video_file):
    # Specify a sub-region for cropping to make the search for the LED easier
    frame_width = 1920
    frame_height = 1080
    led_x0 = 850
    led_x1 = 1160
    led_y0 = 800
    led_y1 = 1080
    #######################################

    ###############################################################################
    # Create a template of the LED when using Blue minus Green mask
    # Expect bright spot surrounded by black
    # Use a blurred circle for the spot to help find the center when matching
    # Start with black rectangle
    led_template = np.zeros((61,61)).astype(np.uint8)
    # Make white circle in the center and 
    #cv2.circle(led_template, (20,20), 10, 255, -1)
    cv2.rectangle(led_template, (5,10), (56,51), 255, -1)
    #led_template = cv2.blur(led_template, (20,20))
    # Offsets from corner to center
    led_template_center_offset_x = 20
    led_template_center_offset_y = 20
    ###############################################################################

    # Define delta for determining the size if the area to use for averaging of LED signal
    # Will average a square +/- delta from center
    led_delta = 3

    # Create data frame for logging info
    df_location  = pd.DataFrame([], columns=['File','FrameNumber', 'CenterX', 'CenterY'])

    # Process the video
    cap = cv2.VideoCapture(path_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Find LED location in several frames across the video
    vertical_composite_list = []
    for frame_location in range(0, total_frames, total_frames//11):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_location)
        ret, frame = cap.read()
        
        f = frame.copy()
        
        # Extract Blue Channel from the specified subregion and threshold it
        led_loc = f[led_y0:led_y1, led_x0:led_x1, :]
        b = f[led_y0:led_y1, led_x0:led_x1, 0]
        g = f[led_y0:led_y1, led_x0:led_x1, 1]
        b_minus_g = cv2.subtract(b, g)
        bg_10 = cv2.blur(b_minus_g, (10,10))
        
        # Find location of the LED block by template matching
        res = cv2.matchTemplate(bg_10, led_template, cv2.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = min_loc
        center = np.add(np.array(top_left), np.array((led_template_center_offset_x, led_template_center_offset_y)))
        print(center)
        
        ##### Annotate a '+' for LED location in cropped image
        led_annotate = led_loc.copy()
        cv2.line(led_annotate, (center[0], center[1]-20), (center[0], center[1]+20), (0,0,255), 1)
        cv2.line(led_annotate, (center[0]-20,  center[1]), (center[0]+20, center[1]), (0,0,255), 1)
        
        # Create image for annotating results
        info = np.zeros((150, led_annotate.shape[1], 3)).astype(np.uint8)
        
        frame_info = 'Frame: ' + str(frame_location)
        cv2.putText(info, frame_info, (10, 35), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA);
        cv2.putText(info, '(row, col)', (10, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA);
        center_info = '(' + str(center[1]) + ', ' + str(center[0]) + ')'
        cv2.putText(info, center_info, (10, 125), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA);
    
        # Create composite to show Crop, Processed image used for search, location
        img1 = cv2.copyMakeBorder(led_loc, 2, 2, 2, 2, cv2.BORDER_CONSTANT, None, value = (127, 127, 127))
        img2 = cv2.copyMakeBorder(cv2.cvtColor(bg_10, cv2.COLOR_BGR2RGB), 2, 2, 2, 2, cv2.BORDER_CONSTANT, None, value = (127, 127, 127))
        img3 = cv2.copyMakeBorder(led_annotate, 2, 2, 2, 2, cv2.BORDER_CONSTANT, None, value = (127, 127, 127))
        img4 = cv2.copyMakeBorder(info, 2, 2, 2, 2, cv2.BORDER_CONSTANT, None, value = (127, 127, 127))
        vertical_composite_list.append(cv2.vconcat([img1, img2, img3, img4]))
    
        df_location_new  = pd.DataFrame([[video_file, frame_location, center[0], center[1]]], columns=['File','FrameNumber', 'CenterX', 'CenterY'])
        df_location = pd.concat([df_location, df_location_new], ignore_index=True)
    
    # Tabulate results for LED location    
    center_x_median = int(df_location['CenterX'].median())
    center_y_median = int(df_location['CenterY'].median())
    center_x_span = int(df_location['CenterX'].max() - df_location['CenterX'].min())
    center_y_span = int(df_location['CenterY'].max() - df_location['CenterY'].min())
    print('Checking location spans:')
    print('X span = ' + str(center_x_span))
    print('Y span = ' + str(center_y_span))
    
    # Adjust location to correspond to the location in the full image
    xy_location = (center[0] + led_x0, center[1] + led_y0)
    df_location['CenterX_FullFrame'] = xy_location[0]
    df_location['CenterY_FullFrame'] = xy_location[1]
    
    ##### Make composite image to summarize LED location findings
    # Compile the results from each frame analyzed
    full_composite = cv2.hconcat(vertical_composite_list)
    
    # Make a header for the summary
    header = np.zeros((150, full_composite.shape[1], 3)).astype(np.uint8)
    summary1 = 'Video File: ' + video_file
    summary2 = 'LED Center (X, Y): (' + str(center_x_median) +', ' +str(center_y_median) +') for Crop, (' + str(xy_location[0]) +', ' + str(xy_location[1]) + ') for Full Image'
    summary3 = 'Span for X: ' + str(center_x_span) + ', ' + 'Span for Y: ' +str (center_y_span)
    cv2.putText(header, summary1, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(header, summary2, (20, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(header, summary3, (20, 125), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    # Combine header and composite
    full_composite = cv2.vconcat([header, full_composite])
    
    # Save summary image and data frame
    # cv2.imwrite(os.path.join(c.output_path, video_file.replace('.MP4', '_LED_Location.PNG')), full_composite)
    # df_location.to_csv(os.path.join(c.output_path, video_file.replace('.MP4', '_LED_Location.csv')), index=False)
    
    # Release video resource
    cap.release()
    
    return xy_location
###############################################################################



###############################################################################
#   Function for extracting the Red LED signal from the video
#   INPUTS:     Path to video file is specified in the AnalysisConfig file
#   OUTPUT:     Dataframe with the Red LED Signal calculated for everey frame of video
def get_alignment_signal_from_video(self, path_video, video_file):
    frame_width = 1920
    frame_height = 1080
    led_x0 = 850
    led_x1 = 1160
    led_y0 = 800
    led_y1 = 1080
    experiment_name = video_file.removesuffix('.mov').removesuffix('.MOV').removesuffix('.MP4').removesuffix('.mp4')
    #######################################

    ###############################################################################
    # Create a template of the LED when using Blue minus Green mask
    # Expect bright spot surrounded by black
    # Use a blurred circle for the spot to help find the center when matching
    # Start with black rectangle
    led_template = np.zeros((61,61)).astype(np.uint8)
    # Make white circle in the center and 
    #cv2.circle(led_template, (20,20), 10, 255, -1)
    cv2.rectangle(led_template, (5,10), (56,51), 255, -1)
    #led_template = cv2.blur(led_template, (20,20))
    # Offsets from corner to center
    led_template_center_offset_x = 20
    led_template_center_offset_y = 20
    ###############################################################################

    # Define delta for determining the size if the area to use for averaging of LED signal
    # Will average a square +/- delta from center
    led_delta = 3

    #### Process each frame of video to get the signals using the pre-determined location for the LED
    df = pd.DataFrame([], columns=['Experiment','FrameNumber', 'RedScore'])
    
    # Get the location of the LED
    center = find_led_location()
    
    # Process the video
    cap = cv2.VideoCapture(path_video)
    
    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_counter%100 == 0:
            print(frame_counter)
        
        # Rotate if top view
        # if c.top_view: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
        # Extract the small portion of the Red Channel for getting the Red Signal
        r = frame[center[1]-led_delta:center[1]+led_delta+1, center[0]-led_delta:center[0]+led_delta+1, 2]
          
        # Calculate the Red Signal by averaging over the small region specified in the config file
        signal_r = np.round(np.mean(r))
        
        # Compile results in the dataframe
    
        df_new = pd.DataFrame([[experiment_name, frame_counter, signal_r]],
                              columns=['Experiment','FrameNumber', 'RedScore'])
    
        df = pd.concat([df, df_new], ignore_index=True)
        frame_counter += 1
    
    # Release resources
    cap.release()
    
    ##### Create a clean Red signal
    red_score_threshold = np.mean([np.percentile(df['RedScore'], 25), np.percentile(df['RedScore'], 75)])
    df['RedScore_Shifted'] = df['RedScore']- red_score_threshold
    df['Video_LED_Signal'] = np.sign(df['RedScore_Shifted'])
    
    # Save data frame with LED signal
    # df.to_csv(os.path.join(c.output_path, video_file.replace('.MOV', '_Video_LED_Signal.csv')), index=False)
    
    # Convert the LED signal into binary ON/OFF
    led = (df['RedScore_Shifted'].to_numpy() > 0).astype(int)
    frames = df['FrameNumber'].to_numpy()

    # Find where the LED switches between ON and OFFs
    edges = np.diff(np.r_[led[0], led])              # changes between samples
    change_idx = np.where(edges != 0)[0]             # indices where state flips
    # Split the LED signal into contiguous ON/OFF blocks
    starts = np.r_[0, change_idx + 1]
    ends   = np.r_[change_idx, len(led) - 1]
    
    starts = np.clip(starts, 0, len(led) - 1)
    ends   = np.clip(ends,   0, len(led) - 1)

    # For each segment, compute length in *frames* (use FrameNumber for robustness)
    seg_is_on = led[starts] == 1
    seg_len_frames = (frames[ends] - frames[starts]) + 1  # inclusive

    # Longest ON and longest OFF (in frames)
    if np.any(seg_is_on):
        longest_on_frames = int(seg_len_frames[seg_is_on].max())
    else:
        raise RuntimeError("No ON segment found in LED signal.")

    if np.any(~seg_is_on):
        longest_off_frames = int(seg_len_frames[~seg_is_on].max())
    else:
        raise RuntimeError("No OFF segment found in LED signal.")

    # Your assumption: longest ON + longest OFF == 0.4 s (Arduino cycle)
    T_cycle_sec = 0.4
    frames_per_cycle = longest_on_frames + longest_off_frames
    actual_fps_est = frames_per_cycle / T_cycle_sec
    print(f"[LED longest] ON={longest_on_frames} frames, OFF={longest_off_frames} frames, "
            f"sum={frames_per_cycle} → actual_fps≈{actual_fps_est:.2f}")
    
    return df

def align_data(self, df_f, df_v, output_folder):
    force_step = 10
    experiment_name = video_file.removesuffix('.mov').removesuffix('.MOV').removesuffix('.MP4').removesuffix('.mp4')


    save_aligned = True
    file_aligned_dataframe = experiment_name + '_{AlignedData}.csv'
    path_aligned_dataframe = os.path.join(parent_path, output_folder, file_aligned_dataframe)

    file_aligned_plot = experiment_name + '_{AlignedData_partial}.png'
    path_aligned_plot = os.path.join(parent_path, output_folder, file_aligned_plot)

    # Create dataframe with subset of force data to match video fps
    df = df_f.iloc[::force_step].reset_index()
    
    # Extract LED signal from force data subset 
    signal_force = df['FP_LED_Signal']
    
    # Extract LED signal from video
    print("Columns in df_video:", df_v.columns.tolist())
    signal_video = df_v['Video_LED_Signal']
    
    # Determine alignment offset
    correlation = signal.correlate(signal_video, signal_force, mode="valid")
    lags = signal.correlation_lags(signal_video.size, signal_force.size, mode="valid")
    lag = lags[np.argmax(correlation)]
    print('Frame Offset for Alignment: ' + str(lag))
    
    max_corr = np.max(correlation)
    perfect_corr = np.min([len(signal_force), len(signal_video)])
    relative_score = max_corr / perfect_corr
    print('Relative Alighnment Score: ' + str(relative_score))

    
    # Create a unified dataframe with the video frame data associated with forces
    df['FrameNumber'] = list(np.arange(lag, lag+len(df)))
    
    df = pd.merge(df, df_v, on='FrameNumber', how='left')
    
    # Show plot
    plt.plot(df['FP_LED_Signal'], alpha=0.50)
    plt.plot(df['Video_LED_Signal'], alpha=0.50, linestyle='dotted')
    plt.xlim((int(0.35*len(df)), int(0.65*len(df))))
    plt.show()

    if save_aligned == True:
        print('Saving aligned data')
        df.to_csv(path_aligned_dataframe)
        
        plt.plot(df['FP_LED_Signal'], alpha=0.50)
        plt.plot(df['Video_LED_Signal'], alpha=0.50, linestyle='dotted')
        plt.xlim((int(0.35*len(df)), int(0.65*len(df))))
        plt.savefig(path_aligned_plot)
    
    return df


def new_led(parent_path, video_file, force_file):
    path_video = os.path.join(parent_path, video_file)
    
    s_time = time.time()
    df_force = process_force_file(parent_path, force_file)
    print('\nTime for process_force_file: ' + str(time.time() - s_time))


    # Process video to extract the alignment signal based on the Red LED
    s_time = time.time()
    df_video = get_alignment_signal_from_video(path_video, video_file)
    print('Time for function: ' + str(time.time() - s_time))


    # Align the force and video data
    s_time = time.time()
    df_aligned = align_data(df_force, df_video)
    print('Time for function: ' + str(time.time() - s_time))

    lagFile = os.path.join(parent_path, '_Results.csv')
    lagValue = df_aligned['Video Frame for t_zero force'].values[0]
    lagValue = int(lagValue)

    return lagValue

def run_led_syncing(self, parent_path, video_file, force_file):
    startTime = time.time()

    # Use the passed parameters instead of hardcoded values
    video_path = os.path.join(parent_path, video_file)

    # --- Template for LED detection ---
    # Template is a rectangle with a cross in the middle
    template = np.zeros((35, 61), dtype=np.uint8)
    template[0:5, :] = 255
    template[30:35, :] = 255
    template[:, 0:5] = 255
    template[:, 56:61] = 255
    template[12:23, 22:39] = 255

    template_center_offset_x = 30 # Offset from the top-left corner of the template to the LED center
    template_center_offset_y = 17 # Offset from the top-left corner of the template to the LED center
    delta = 3  # Area around LED center for signal averaging

    # --- Open video ---
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        raise Exception("Failed to read first frame.")

    # --- Find initial LED center (using blue channel) ---
    b_first = first_frame[:, :, 0]
    # Threshold full image for visualization and ROI for detection (bottom half only)
    _, thresh_b_first_full = cv2.threshold(b_first, 127, 255, cv2.THRESH_BINARY)
    h_first, w_first = b_first.shape
    roi_y0_first = h_first // 2
    _, thresh_b_first_roi = cv2.threshold(b_first[roi_y0_first:, :], 127, 255, cv2.THRESH_BINARY)
    # Template matching on bottom half ROI
    res = cv2.matchTemplate(thresh_b_first_roi, template, cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # Map ROI coordinates back to full-frame
    top_left = (min_loc[0], min_loc[1] + roi_y0_first)
    led_center = np.add(top_left, (template_center_offset_x, template_center_offset_y))
    cv2.drawMarker(first_frame, top_left, (255, 0, 0), thickness=3)
    cv2.drawMarker(first_frame, led_center, (255, 0, 0), thickness=3)
    cv2.namedWindow("Detected LED", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detected LED", 800, 600)  # Set window size
    cv2.imshow("Detected LED", first_frame)
    cv2.waitKey(0)

    # --- Prepare output ---
    df = pd.DataFrame(columns=['File', 'FrameNumber', 'CenterX', 'CenterY', 'BlueScore', 'GreenScore', 'RedScore'])
    frame_counter = 0

    # --- Loop through video ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]

        # Search only in bottom half of the frame
        h, w = b.shape
        roi_y0 = h // 2
        _, thresh_b_roi = cv2.threshold(b[roi_y0:, :], 127, 255, cv2.THRESH_BINARY)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = (min_loc[0], min_loc[1] + roi_y0)
        led_center = np.add(top_left, (template_center_offset_x, template_center_offset_y))
        if frame_counter < 5:
            cv2.drawMarker(frame, top_left, (255, 0, 0), thickness=3)
            cv2.drawMarker(frame, led_center, (255, 0, 0), thickness=3)
            cv2.namedWindow("Detected LED", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detected LED", 800, 600)  # Set window size
            cv2.imshow("Detected LED", frame)
            cv2.waitKey(0)
        y, x = int(led_center[1]), int(led_center[0])
        y0 = max(0, y - delta)
        y1 = min(b.shape[0], y + delta + 1)
        x0 = max(0, x - delta)
        x1 = min(b.shape[1], x + delta + 1)
 
        signal_b = np.round(np.mean(b[y0:y1, x0:x1]))
        signal_g = np.round(np.mean(g[y0:y1, x0:x1]))
        signal_r = np.round(np.mean(r[y0:y1, x0:x1]))

        df_new = pd.DataFrame([[video_file, frame_counter, x, y, signal_b, signal_g, signal_r]],
                              columns=df.columns)
        df = pd.concat([df, df_new], ignore_index=True)
        frame_counter += 1

    cap.release()

    # --- Clean red signal ---
    red_score_threshold = np.mean([np.percentile(df['RedScore'], 25), np.percentile(df['RedScore'], 75)])
    df['RedScore_Shifted'] = df['RedScore'] - red_score_threshold
    df['RedScore_Clean'] = np.sign(df['RedScore_Shifted'])

    # --- Find Actual FPS ---
    led = (df['RedScore_Clean'].to_numpy() > 0).astype(int)
    frames = df['FrameNumber'].to_numpy()

    # Find contiguous ON and OFF segments
    edges = np.diff(np.r_[led[0], led])              # changes between samples
    change_idx = np.where(edges != 0)[0]             # indices where state flips
    # segment boundaries (start idx inclusive, end idx inclusive)
    starts = np.r_[0, change_idx + 1]
    ends   = np.r_[change_idx, len(led) - 1]

    # For each segment, compute length in *frames* (use FrameNumber for robustness)
    seg_is_on = led[starts] == 1
    seg_len_frames = (frames[ends] - frames[starts]) + 1  # inclusive

    # Longest ON and longest OFF (in frames)
    if np.any(seg_is_on):
        longest_on_frames = int(seg_len_frames[seg_is_on].max())
    else:
        raise RuntimeError("No ON segment found in LED signal.")

    if np.any(~seg_is_on):
        longest_off_frames = int(seg_len_frames[~seg_is_on].max())
    else:
        raise RuntimeError("No OFF segment found in LED signal.")

    # Your assumption: longest ON + longest OFF == 0.4 s (Arduino cycle)
    T_cycle_sec = 0.4
    frames_per_cycle = longest_on_frames + longest_off_frames
    actual_fps_est = frames_per_cycle / T_cycle_sec

    # --- Save video analysis ---
    df_filename = video_file.replace('.mov', '_Analysis_Front.csv').replace('.MOV', '_Analysis_Front.csv')
    df.to_csv(os.path.join(parent_path, df_filename), index=False)

    # --- Load and process force plate data ---
    force_path = os.path.join(parent_path, force_file)
    df_force = pd.read_csv(force_path, header=17, delimiter='\t', encoding='latin1').drop(0)

    df_force['RedSignal'] = np.sign(df_force['Fz.2'].astype('float64'))

    # Downscaling by 10
    df_force_subset = df_force.iloc[::10].reset_index(drop=True) #temporary
    signal_force = df_force_subset['RedSignal']
    signal_video = df['RedScore_Clean']

    # --- Align signals (z-normalized) ---
    # Convert to float and z-normalize to mitigate imbalance/offsets
    video_arr = np.asarray(signal_video, dtype=float)
    force_arr = np.asarray(signal_force, dtype=float)
    if np.std(video_arr) > 0:
        video_arr = (video_arr - np.mean(video_arr)) / np.std(video_arr)
    else:
        video_arr = video_arr - np.mean(video_arr)
    if np.std(force_arr) > 0:
        force_arr = (force_arr - np.mean(force_arr)) / np.std(force_arr)
    else:
        force_arr = force_arr - np.mean(force_arr)

    correlation = signal.correlate(video_arr, force_arr, mode="valid")
    lags = signal.correlation_lags(video_arr.size, force_arr.size, mode="valid")
    lag = lags[np.argmax(correlation)]

    # --- Save aligned force data ---
    df_force_filename = force_file.replace('.txt', '_Analysis_Force.csv')
    df_force_subset.to_csv(os.path.join(parent_path, df_force_filename), index=False)

    print(f"Saved force data to file_path: {os.path.join(parent_path, df_force_filename)}")

    # --- Save final alignment result ---
    max_corr = float(np.max(correlation))
    perfect_corr = min(len(force_arr), len(video_arr))
    relative_score = max_corr / perfect_corr

    df_result = pd.DataFrame([[video_file, force_file, lag, max_corr, perfect_corr, relative_score]],
                             columns=['Video File', 'Force File', 'Video Frame for t_zero force',
                                      'Correlation Score', 'Perfect Score', 'Relative Score'])

    df_result_filename = force_file.replace('.txt', '_Results.csv')
    df_result.to_csv(os.path.join(parent_path, df_result_filename), index=False)

    print(f"Done. Columns in force data: {df_force.columns.tolist()}")
    print(f"[LED longest] ON={longest_on_frames} frames, OFF={longest_off_frames} frames, "
        f"sum={frames_per_cycle} → actual_fps≈{actual_fps_est:.2f}")
    print(f"[DEBUG] The relative score is {relative_score}")

    if lag >= 0:
        force_pad = pd.concat([pd.Series(np.zeros(lag)), df_force_subset['RedSignal']], ignore_index=True)
    else:
        force_pad = df_force_subset['RedSignal'].iloc[abs(lag):].reset_index(drop=True)

    n = min(len(signal_video), len(force_pad))
    sig = np.asarray(signal_video[:n])
    frc = np.asarray(force_pad[:n])

    # # --- Plot (same styling & axes as your original) ---
    # from matplotlib.figure import Figure
    # from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    # fig = Figure(figsize=(8, 4))
    # canvas = FigureCanvas(fig)
    # ax = fig.add_subplot(111)

    # ax.step(sig, alpha=0.5, label="Video signal")                    # same alpha
    # ax.step(frc, alpha=0.5, label="Force")   # dashed line
    # ax.plot()
    # # same title/xlabel/xlim as your pyplot code
    # ax.set_title(f"Alignment Using a Lag of {lag} Frames")
    # ax.set_xlabel("Frame ID")
    # print(f"[DEBUG] Total frames: {self.Video.total_frames}")
    # # n = len(sig)
    # ax.set_xlim(n//2 - 500, n//2 + 500)

    # ax.legend(loc="best")
    # fig.tight_layout()
    # fig.savefig(os.path.join(parent_path, "led_sync_preview.png"), dpi=150, bbox_inches="tight")
    # --- Plot (same styling & axes as your original) ---
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    fig = Figure(figsize=(8, 4))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    # Explicit x-values prevent step() from misinterpreting arguments
    x = np.arange(n)

    ax.step(x, sig, where='mid', alpha=0.5, label="Video signal")
    ax.step(x, frc, where='mid', alpha=0.5, label="Force")

    # Labels and title
    ax.set_title(f"Alignment Using a Lag of {lag} Frames")
    ax.set_xlabel("Frame ID")

    # X-axis limits (safe even when n < 1000)
    # left = max(0, n//2 - 500)
    # right = min(n, n//2 + 500)
    ax.set_xlim(0, n)

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(parent_path, "led_sync_preview.png"),
                dpi=150, bbox_inches="tight")

    lagFile = os.path.join(parent_path, '_Results.csv')
    lagValue = df_result['Video Frame for t_zero force'].values[0]
    lagValue = int(lagValue)

    return lagValue

# Allow the script to be run directly if needed
if __name__ == "__main__":
    # Default values for direct execution
    parent_path = r"C:\Users\Deren\OneDrive\Desktop\USCProject\Prepare2Play-Learning-Experience\newData"
    video_file = "walk_test_vid01.mov"
    force_file = "walktest1.txt"
    run_led_syncing(parent_path, video_file, force_file)