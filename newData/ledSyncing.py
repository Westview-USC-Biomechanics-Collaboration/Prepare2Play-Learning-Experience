import cv2
import numpy as np
import pandas as pd
import time
import os
from scipy import signal

def run_led_syncing(parent_path, video_file, force_file):
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

        
        res = cv2.matchTemplate(thresh_b_roi, template, cv2.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = (min_loc[0], min_loc[1] + roi_y0)
        led_center = np.add(top_left, (template_center_offset_x, template_center_offset_y))
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

    # --- Save video analysis ---
    df_filename = video_file.replace('.mov', '_Analysis_Front.csv').replace('.MOV', '_Analysis_Front.csv')
    df.to_csv(os.path.join(parent_path, df_filename), index=False)

    # --- Load and process force plate data ---
    force_path = os.path.join(parent_path, force_file)
    df_force = pd.read_csv(force_path, header=17, delimiter='\t', encoding='latin1').drop(0)

    df_force['RedSignal'] = np.sign(df_force['Fz.2'].astype('float64'))

    # Downsample force to match video rate (10x slower)
    df_force_subset = df_force.iloc[::10].reset_index(drop=True)
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