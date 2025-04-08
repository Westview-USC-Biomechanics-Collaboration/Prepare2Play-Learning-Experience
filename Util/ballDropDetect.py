"""
This is a utility script for ball drop detect
args:
    cap: video capture object (open cv)
    
return:
    index: the index of the tennis ball drop
"""
import cv2
import numpy as np
import time
import pandas as pd


def ballDropDetect(cap:cv2.VideoCapture):
    # record startTime
    startTime = time.time()

    # Define color range for darker green tennis ball (adjust based on lighting conditions)
    lower_bound = np.array([25, 130, 50])  # Darker green lower bound in HSV
    upper_bound = np.array([80, 240, 140])  # Upper bound in HSV (limit brightness)
    MIN_AREA_THRESHOLD = 500  

    # Initialize variables
    frame_index = 0
    lowPixel = []

    index:int = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for tennis ball color
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Filter out small contours based on area
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA_THRESHOLD]

            if valid_contours:
                # Find the largest valid contour by area
                largest_contour = max(valid_contours, key=cv2.contourArea)
                
                # Find the lowest point in the contour
                lowest_point = max(largest_contour, key=lambda p: p[0][1])  # p[0][1] is the y-coordinate
                
                # Store frame index and lowest position
                lowPixel.append({"index": frame_index, "position": int(lowest_point[0][1])})
        
        frame_index +=1
        # End of loop

    if lowPixel:
        # find lowest pixel
        highest_position_entry = max(lowPixel, key=lambda x: x["position"])
        
        print(highest_position_entry)
        # find proccessing time
        print(f"Process time: {time.time()-startTime} s")
        
        # print result
        print(f"The ball reaches its lowest position at frame index: {highest_position_entry['index']}")

    return highest_position_entry['index']

def forceSpikeDetect(df:pd.DataFrame):
    init_force = df.loc[0,"Fz2"]
    target_row = -1
    for index, f in enumerate(df.loc[:,"Fz2"]):
        if((f-init_force)>30):
            target_row = index
            break

    return target_row


if __name__ == "__main__":
    capture = cv2.VideoCapture('/home/chaser/Downloads/tss_rl_JG_vid02.mov')
    index = ballDropDetect(capture)
    print("program finished")