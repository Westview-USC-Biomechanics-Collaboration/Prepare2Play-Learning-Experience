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


def ballDropDetect(cap: cv2.VideoCapture):
    import time
    import numpy as np
    import cv2

    startTime = time.time()

    lower_bound = np.array([25, 130, 50])    # HSV lower bound for green
    upper_bound = np.array([80, 240, 140])   # HSV upper bound for green
    MIN_AREA_THRESHOLD = 500

    frame_index = 0
    lowPixel = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA_THRESHOLD]
            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                lowest_point = max(largest_contour, key=lambda p: p[0][1])
                lowPixel.append({"index": frame_index, "position": int(lowest_point[0][1])})

        frame_index += 1

    # Always define `highest_position_entry`, whether or not lowPixel has data
    if lowPixel:
        highest_position_entry = max(lowPixel, key=lambda x: x["position"])
        print(f"Lowest point detected at frame {highest_position_entry['index']}, pixel y = {highest_position_entry['position']}")
    else:
        highest_position_entry = {"index": 0, "position": -1}
        print("No ball detected.")

    print(f"Process time: {time.time() - startTime:.2f} s")

    return highest_position_entry['index']



def forceSpikeDetect(df:pd.DataFrame):
    init_force = df.loc[0,"Fz2"]
    target_row = 0
    for index, f in enumerate(df.loc[:,"Fz2"]):
        if((f-init_force)>30):
            target_row = index
            break

    return target_row


if __name__ == "__main__":
    capture = cv2.VideoCapture('/home/chaser/Downloads/tss_rl_JG_vid02.mov')
    index = ballDropDetect(capture)
    print("program finished")