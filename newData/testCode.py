# -*- coding: utf-8 -*-
"""
Created on Sun May 11 21:29:02 2025

@author: srong
"""

import cv2
import numpy as np
import pandas as pd
import time
import os
import glob
from matplotlib import pyplot as plt
from scipy import signal

startTime = time.time()

######### Alignment of Force data to Front Video

parent_path = r"C:\Users\gulbd\OneDrive\Documents\GitHub\Prepare2Play-Learning-Experience\newData"
video_file = "tennisball_test_long_vid03.MOV"
video_path = os.path.join(parent_path, video_file)

# Load video
cap = cv2.VideoCapture(video_path)


################## Create a template of the LED block #########################
# Assume black block, with small white border and small white center region
# Start with black rectangle
template = np.zeros((35,61)).astype(np.uint8)
# Make border white
template[0:5, 0:61] = 255
template[30:35, 0:61] = 255
template[0:35, 0:5] = 255
template[0:35, 56:61] = 255
# Make a white center
template[12:23, 22:39] = 255
template_center_offset_x = 30
template_center_offset_y = 17

# Define delta for determining the size if the area to use for averaging of LED signal
# Will average a square +/- delta from center
delta = 3
###############################################################################


df = pd.DataFrame([], columns=['File','FrameNumber', 'CenterX', 'CenterY', 'BlueScore', 'GreenScore', 'RedScore'])


frame_counter = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_counter%100 == 0:
        print(frame_counter)

    b = frame[:, :, 0]
    g = frame[:, :, 1]
    r = frame[:, :, 2]
    
    ret, thresh_b = cv2.threshold(b, 127, 255, cv2.THRESH_BINARY)
    ret, thresh_g = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
    ret, thresh_r = cv2.threshold(r, 127, 255, cv2.THRESH_BINARY)
    
    # Find center of LED block by template matching with the Blue Channel
    res = cv2.matchTemplate(thresh_b,template,cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc
    center = np.add(np.array(top_left), np.array((template_center_offset_x, template_center_offset_y)))
                  
    # Calculate the signal using the Red Channel for a small region around the center of the loaction found for the block
    signal_b = np.round(np.mean(b[center[1]-delta:center[1]+delta+1, center[0]-delta:center[0]+delta+1]))
    signal_g = np.round(np.mean(g[center[1]-delta:center[1]+delta+1, center[0]-delta:center[0]+delta+1]))
    signal_r = np.round(np.mean(r[center[1]-delta:center[1]+delta+1, center[0]-delta:center[0]+delta+1]))

    df_new = pd.DataFrame([[video_file, frame_counter, center[0], center[1], signal_b, signal_g, signal_r]],
                          columns=['File','FrameNumber', 'CenterX', 'CenterY', 'BlueScore', 'GreenScore', 'RedScore'])

    df = pd.concat([df, df_new], ignore_index=True)
    
    frame_counter += 1


# Release resources
cap.release()



##### Create a clean Red signal
red_score_threshold = np.mean([np.percentile(df['RedScore'], 25), np.percentile(df['RedScore'], 75)])

df['RedScore_Shifted'] = df['RedScore']- red_score_threshold
df['RedScore_Clean'] = np.sign(df['RedScore_Shifted'])


# Save analysis
df_filename = video_file.replace('.mov', '_Analysis_Front.csv').replace('.MOV', '_Analysis_Front.csv')
df.to_csv(os.path.join(parent_path, df_filename))





#########   Get force plate data
force_file = "tennisball_test_for03.txt"
force_path = os.path.join(parent_path, force_file)

df_force = pd.read_csv(force_path, header=17, delimiter='\t')
df_force = df_force.drop(0)

df_force['RedSignal'] = np.sign(df_force['Fz.2'].astype('float64') )

# Create subset using everey 1oth row since DAQ sampling is 10X front video
df_force_subset = df_force.iloc[::10].reset_index()

# Extract the two signals we want to align
signal_force = df_force_subset['RedSignal']
signal_video = df['RedScore_Clean']

# Determin alignment offset
correlation = signal.correlate(signal_video, signal_force, mode="valid")
lags = signal.correlation_lags(signal_video.size, signal_force.size, mode="valid")
lag = lags[np.argmax(correlation)]


df_filename = force_file.replace('.txt', '_Analysis_Force.csv')
df_force_subset.to_csv(os.path.join(parent_path, df_filename))

max_corr = np.max(correlation)
perfect_corr = np.min([len(signal_force), len(signal_video)])


df_result = pd.DataFrame([[video_file, force_file, lag, max_corr, perfect_corr, max_corr/perfect_corr]],
                         columns=['Video File', 'Force File', 'Video Frame for t_zero force', 'Correlation Score', 'Perfect Score', 'Relative Score'])


df_filename = force_file.replace('.txt', '_Results.csv')
df_result.to_csv(os.path.join(parent_path, df_filename))

print(f"Colunms : {df_force.columns}")