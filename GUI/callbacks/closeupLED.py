"""
Standalone LED Close-Up Signal Extractor

Uses the same methods as the main LED detection system:
- Blue dominance processing (blue - green, blue - red)
- Template matching (TM_CCOEFF_NORMED)
- Red channel signal extraction
- Binary thresholding via 25th/75th percentile

For a close-up video where the entire frame is the LED — no crop needed.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def process_crop_for_matching(frame: np.ndarray, blur_kernel: int = 10) -> np.ndarray:
    """
    Same blue-dominance processing used in the main system.
    Pixel must beat BOTH green and red channels to register — suppresses shadows.
    """
    blue_channel  = frame[:, :, 0]
    green_channel = frame[:, :, 1]
    red_channel   = frame[:, :, 2]

    blue_minus_green = cv2.subtract(blue_channel, green_channel)
    blue_minus_red   = cv2.subtract(blue_channel, red_channel)

    blue_dominant = cv2.min(blue_minus_green, blue_minus_red)
    processed = cv2.blur(blue_dominant, (blur_kernel, blur_kernel))

    return processed


def create_led_template(indoor: bool = True) -> np.ndarray:
    """
    Same templates used in the main system.
    indoor=True  → rectangular block template (indoor LED box)
    indoor=False → circular glow template (outdoor LED dot)
    """
    if indoor:
        template = np.zeros((71, 91), dtype=np.uint8)
        cv2.rectangle(template, (20, 27), (71, 68), 200, -1)
        cv2.rectangle(template, (42, 30), (49, 45), 10, -1)
        template = cv2.blur(template, (5, 5))
    else:
        template = np.zeros((31, 31), dtype=np.uint8)
        center = (15, 15)
        cv2.circle(template, center, 10,  60, -1)
        cv2.circle(template, center,  6, 150, -1)
        cv2.circle(template, center,  3, 255, -1)
        template = cv2.blur(template, (5, 5))

    return template


def find_led_location_closeup(
    video_path: str,
    indoor: bool = True,
    num_frames_to_check: int = 11,
    match_threshold: float = 0.3,
    blur_kernel: int = 10,
) -> tuple:
    """
    Find LED center in close-up video using template matching.
    No crop — entire frame is the search region.

    Returns (median_x, median_y) in frame coordinates.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    template = create_led_template(indoor=indoor)

    frame_indices = np.linspace(0, total_frames - 1, num_frames_to_check, dtype=int)
    centers_x, centers_y, scores = [], [], []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        processed = process_crop_for_matching(frame, blur_kernel=blur_kernel)
        result = cv2.matchTemplate(processed, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < match_threshold:
            print(f"  Frame {idx}: low confidence ({max_val:.2f}), skipping")
            continue

        cx = max_loc[0] + template.shape[1] // 2
        cy = max_loc[1] + template.shape[0] // 2
        centers_x.append(cx)
        centers_y.append(cy)
        scores.append(max_val)
        print(f"  Frame {idx}: center=({cx}, {cy}), score={max_val:.3f}")

    cap.release()

    if not centers_x:
        raise RuntimeError("LED not detected in any sampled frame. Try lowering match_threshold or switching indoor/outdoor mode.")

    median_x = int(np.median(centers_x))
    median_y = int(np.median(centers_y))
    print(f"\nMedian LED center: ({median_x}, {median_y})")
    print(f"X span: {max(centers_x)-min(centers_x)}px, Y span: {max(centers_y)-min(centers_y)}px")

    return median_x, median_y


def extract_led_signal_closeup(
    video_path: str,
    led_center: tuple,
    signal_delta: int = 5,
) -> pd.DataFrame:
    """
    Extract Red channel signal from every frame around the LED center.
    Same method as main system: average Red in (center ± delta) window,
    then threshold via mean of 25th/75th percentiles.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    cx, cy = led_center
    signals = []
    frame_counter = 0

    print(f"\nExtracting Red channel signal...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_counter % 200 == 0:
            print(f"  Frame {frame_counter}...")

        red_region = frame[
            cy - signal_delta : cy + signal_delta + 1,
            cx - signal_delta : cx + signal_delta + 1,
            2  # Red channel (BGR ordering)
        ]
        red_score = float(np.mean(red_region))
        signals.append({'FrameNumber': frame_counter, 'RedScore': red_score})
        frame_counter += 1

    cap.release()

    df = pd.DataFrame(signals)

    # Same thresholding as main system
    threshold = np.mean([
        np.percentile(df['RedScore'], 25),
        np.percentile(df['RedScore'], 75),
    ])
    df['RedScore_Shifted'] = df['RedScore'] - threshold
    df['Video_LED_Signal'] = np.sign(df['RedScore_Shifted'])

    print(f"  Total frames: {len(df)}")
    print(f"  Signal threshold: {threshold:.1f}")
    print(f"  LED ON frames:  {(df['Video_LED_Signal'] > 0).sum()}")
    print(f"  LED OFF frames: {(df['Video_LED_Signal'] <= 0).sum()}")

    return df


def plot_led_signal(df: pd.DataFrame, output_path: str = None, fps: float = None):
    """
    Plot raw Red score alongside the binary digital signal.
    """
    frames = df['FrameNumber'].values
    x_axis = frames / fps if fps else frames
    x_label = 'Time (s)' if fps else 'Frame Number'

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle('LED Close-Up Signal Extraction', fontsize=14, fontweight='bold')

    # --- Raw Red score ---
    axes[0].plot(x_axis, df['RedScore'], color='tomato', linewidth=0.8, label='Raw Red Score')
    threshold = np.mean([
        np.percentile(df['RedScore'], 25),
        np.percentile(df['RedScore'], 75),
    ])
    axes[0].axhline(threshold, color='black', linestyle='--', linewidth=1, label=f'Threshold ({threshold:.1f})')
    axes[0].set_ylabel('Red Intensity', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- Binary digital signal ---
    axes[1].step(x_axis, df['Video_LED_Signal'], color='steelblue', linewidth=1.2, where='post', label='Digital Signal')
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_yticklabels(['OFF (-1)', '0', 'ON (+1)'])
    axes[1].set_ylabel('LED State', fontsize=11)
    axes[1].set_xlabel(x_label, fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")

    plt.show()
    return fig


def run_led_closeup(
    video_path: str,
    output_dir: str = None,
    indoor: bool = True,
    num_frames_to_check: int = 11,
    match_threshold: float = 0.3,
    blur_kernel: int = 10,
    signal_delta: int = 5,
    fps: float = None,
) -> pd.DataFrame:
    """
    Full pipeline for a close-up LED video:
      1. Find LED center via template matching
      2. Extract Red channel signal from every frame
      3. Threshold → binary digital signal
      4. Plot raw + digital signal
      5. Save CSV (if output_dir provided)

    Args:
        video_path:           Path to the close-up LED video
        output_dir:           Directory to save CSV and plot (optional)
        indoor:               True = rectangular block template, False = circular dot template
        num_frames_to_check:  Frames sampled for LED location detection
        match_threshold:      Min TM_CCOEFF_NORMED score to accept a detection (0–1)
        blur_kernel:          Blur size for blue-dominance processing (must be odd or even — cv2.blur accepts both)
        signal_delta:         Half-size of Red extraction window in pixels
        fps:                  Video FPS for time-axis on plot (auto-detected if None)

    Returns:
        DataFrame with columns: FrameNumber, RedScore, RedScore_Shifted, Video_LED_Signal
    """
    print("=" * 60)
    print("LED Close-Up Signal Extractor")
    print(f"  Video:    {video_path}")
    print(f"  Mode:     {'Indoor (block template)' if indoor else 'Outdoor (dot template)'}")
    print("=" * 60)

    # Auto-detect FPS if not supplied
    if fps is None:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        cap.release()
        if fps:
            print(f"  Auto-detected FPS: {fps:.2f}")

    # Step 1: locate LED
    led_center = find_led_location_closeup(
        video_path,
        indoor=indoor,
        num_frames_to_check=num_frames_to_check,
        match_threshold=match_threshold,
        blur_kernel=blur_kernel,
    )

    # Step 2: extract signal
    df = extract_led_signal_closeup(video_path, led_center, signal_delta=signal_delta)

    # Step 3: save + plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "LED_Closeup_Signal.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nCSV saved to: {csv_path}")
        plot_path = os.path.join(output_dir, "LED_Closeup_Signal.png")
    else:
        plot_path = None

    plot_led_signal(df, output_path=plot_path, fps=fps)

    return df


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    VIDEO_PATH  = r"C:/Users/outre/Downloads/20260303_180236_{Day3standing4}_Top.MP4"   # <-- change this
    OUTPUT_DIR  = r"downloads"            # <-- change this (or set None)

    df = run_led_closeup(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        indoor=True,          # switch to False for the outdoor circular LED
        match_threshold=0.3,  # lower if LED isn't being detected
        signal_delta=5,       # pixel radius around LED center for Red averaging
    )

    print("\nFirst 10 rows:")
    print(df.head(10))