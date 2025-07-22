"""

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

parent_path = r"C:\Users\Deren\OneDrive\Desktop\USCProject\Prepare2Play-Learning-Experience\newData"
video_file = "pbd_IT_12.vid04.MOV"
frame_folder = "Frames"
video_path = os.path.join(parent_path, video_file)

force_file = "pbd_IT_12.for04.txt"
force_path = os.path.join(parent_path, force_file)

# Load video
cap = cv2.VideoCapture(video_path)

#crop_left, crop_right, crop_top, crop_bottom = 100, 140, 1480, 1560

# This section can be uncommented to save every frame as an image to a folder
# frame_counter = 0
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     # if frame_counter > 100:
#     #     break

#     # Crop subregion
#     #crop = frame[crop_top:crop_bottom, crop_left:crop_right, :]

#     # Save frame as image
#     frameID = '_' + str(frame_counter).zfill(4) + '.png'
#     frame_name = video_file.replace('.mov', frameID).replace('.MOV', frameID)
#     frame_path = os.path.join(parent_path, frame_folder, frame_name)
#     cv2.imwrite(frame_path, frame)
#     print(frame_name)
#     frame_counter += 1


# # Release resources
cap.release()


#### Create a template of the LED block
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


# cv2.imshow('Template',template)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#### Now work on frames from store files


file_list = glob.glob(os.path.join(parent_path, frame_folder, "*.png"))



current_file = file_list[30]

df = pd.DataFrame([], columns=['File','FrameNumber', 'CenterX', 'CenterY', 'BlueScore', 'GreenScore', 'RedScore'])

for current_file in file_list:
    current_name = current_file.split('\\')[-1]
    current_frame = int(current_name.split('_')[-1].split('.png')[0])
    if current_frame%100 == 0:
        print(current_frame)

    bgr = cv2.imread(current_file)
    b = bgr[:, :, 0]
    g = bgr[:, :, 1]
    r = bgr[:, :, 2]

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

    df_new = pd.DataFrame([[current_name, current_frame, center[0], center[1], signal_b, signal_g, signal_r]],
                          columns=['File','FrameNumber', 'CenterX', 'CenterY', 'BlueScore', 'GreenScore', 'RedScore'])

    df = pd.concat([df, df_new], ignore_index=True)


# Save analysis
df_filename = video_file.replace('.mov', '_Analysis.csv').replace('.MOV', '_Analysis.csv')
df.to_csv(os.path.join(parent_path, df_filename))



##### Create a clean Red signal
# Calculate percentiles
print(np.percentile(df['RedScore'], 0))
print(np.percentile(df['RedScore'], 10))
print(np.percentile(df['RedScore'], 20))
print(np.percentile(df['RedScore'], 30))
print(np.percentile(df['RedScore'], 40))
print(np.percentile(df['RedScore'], 50))
print(np.percentile(df['RedScore'], 60))
print(np.percentile(df['RedScore'], 70))
print(np.percentile(df['RedScore'], 80))
print(np.percentile(df['RedScore'], 90))
print(np.percentile(df['RedScore'], 100))
red_score_threshold = np.mean([np.percentile(df['RedScore'], 25), np.percentile(df['RedScore'], 75)])



df['RedScore_Shifted'] = df['RedScore']- red_score_threshold
df['RedScore_Clean'] = np.sign(df['RedScore_Shifted'])



# Save analysis
df_filename = video_file.replace('.mov', '_Analysis2.csv').replace('.MOV', '_Analysis2.csv')
df.to_csv(os.path.join(parent_path, df_filename))





#########   Get force plate data

df_force = pd.read_csv(force_path, header=17, delimiter='\t')
df_force = df_force.drop(0)
df_force = df_force.astype(float)

df_force['RedSignal'] = np.sign(df_force['Fz.2'].astype('float64') )


# Create subset using every nth row since DAQ sampling is n times the video frame rate
# Comment/uncomment the desired subsampling case
#df_force_subset = df_force.iloc[::10].reset_index() # Force data at 10x video fps (e.g. 2400 & 240)
df_force_subset = df_force.iloc[::5].reset_index()  # Force data at  5x video fps (e.g. 1200 & 240)



plt.plot(df_force_subset['abs time (s)'], df_force_subset['RedSignal'])
plt.show()


# Extract the two signals we want to align
signal_force = df_force_subset['RedSignal']
signal_video = df['RedScore_Clean']


# correlation = signal.correlate(signal_force, signal_video, mode="valid")
# lags = signal.correlation_lags(signal_force.size, signal_video.size, mode="valid")
# lag = lags[np.argmax(correlation)]



correlation = signal.correlate(signal_video, signal_force, mode="valid")
lags = signal.correlation_lags(signal_video.size, signal_force.size, mode="valid")
lag = lags[np.argmax(correlation)]

max_corr = np.max(correlation)
perfect_corr = np.min([len(signal_force), len(signal_video)])
relative_score = max_corr / perfect_corr

plt.plot(correlation)
plt.show()

# Save signal alignment files
df_filename = video_file.replace('.mov', '_Analysis.csv').replace('.MOV', '_Alignment_Video.csv')
df.to_csv(os.path.join(parent_path, df_filename))

df_filename = video_file.replace('.mov', '_Analysis.csv').replace('.MOV', '_Alignment_Force.csv')
df_force_subset.to_csv(os.path.join(parent_path, df_filename))



delay = lag

print(f"Delay: {delay} frames")

plt.plot(signal_video)
plt.xlim(delay, delay+len(signal_force))
plt.show()

plt.plot(signal_force)
plt.show()




##### Create a simple annotated video
#   1) Will trim the video to just the frames with force data
#   2) Will annotate the Fz values (numerically) on each frame
#
#


# Need to load the video again
cap = cv2.VideoCapture(video_path)

# Get info about the video
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Specify the codec to use
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

##### Create video writer for annotated video
# Use 10 fps for playback to give 24X slow motion
annotated_file = video_file.replace('.MOV', '_{annotated_30fps}.mp4')
annotated_path = os.path.join(parent_path, annotated_file)
out = cv2.VideoWriter(annotated_path, fourcc, 30, (frame_width, frame_height))

# Skip forward to account of lag
cap.set(cv2.CAP_PROP_POS_FRAMES, lag)
frame_counter = 0


# Loop through rows of force data
# Note, relying on the idea that video stream extends past the force data
for index, row in df_force_subset.iterrows():
    # Read current frame
    ret, frame = cap.read()

    if frame_counter%100 == 0:
        print(frame_counter)

    # Get Fz for both force plates, round to integer
    Fz_1 = int(row['Fz'])
    Fz_2 = int(row['Fz.1'])

    # Annotate frame
    cv2.putText(frame, 'Fz_1: ' + str(Fz_1), (500, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, 'Fz_2: ' + str(Fz_2), (1300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Write annotated frame to output stream
    out.write(frame)
    frame_counter += 1


# lag_df = pd.DataFrame({"Lag": [lag]})
# lag_filename = video_file.replace('.MOV', '_Lag.csv').replace('.mov', '_Lag.csv')
# lag_df.to_csv(os.path.join(parent_path, lag_filename), index=False)

# Done processing video, so release resources
cap.release()
out.release()

#Create a .csv file with the lag value




#-----------------------------------------------------





















