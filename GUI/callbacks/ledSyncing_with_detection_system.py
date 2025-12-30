"""
Enhanced LED Syncing with led_detection_system.py integration.
This uses view-specific LED detection for more robust alignment.
"""
import cv2
import numpy as np
import pandas as pd
import os
from scipy import signal
import matplotlib.pyplot as plt

# Import the new LED detection system
from GUI.callbacks.led_detection_system import (
    process_view,
    LongViewLEDConfig,
    TopViewLEDConfig,
    SideViewLEDConfig,
    LEDDetector
)

# Keep original functions for backward compatibility
from GUI.callbacks.ledSyncing import (
    plate_transformation_matrix,
    process_force_file,
    align_data
)

def swap_force_plates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Swap FP1 and FP2 data in the dataframe.
    
    Args:
        df: DataFrame with FP1_* and FP2_* columns
        
    Returns:
        DataFrame with FP1 and FP2 data swapped
    """
    df = df.copy()
    
    # Define column pairs to swap
    swap_pairs = [
        ('FP1_Fx', 'FP2_Fx'),
        ('FP1_Fy', 'FP2_Fy'),
        ('FP1_Fz', 'FP2_Fz'),
        ('FP1_|F|', 'FP2_|F|'),
        ('FP1_Ax', 'FP2_Ax'),
        ('FP1_Ay', 'FP2_Ay'),
    ]
    
    for fp1_col, fp2_col in swap_pairs:
        if fp1_col in df.columns and fp2_col in df.columns:
            # Swap the columns
            df[fp1_col], df[fp2_col] = df[fp2_col].copy(), df[fp1_col].copy()
    
    print("[SWAP] Swapped FP1 ↔ FP2 data")
    return df

def new_led_with_detection_system(self, view, parent_path, video_file, force_file):
    """
    Enhanced LED alignment using the detection system for robust LED location finding.
    
    This replaces the old template matching with view-specific detection.
    
    Args:
        self: GUI instance
        view: "Long View", "Top View", or "Short View"
        parent_path: Directory containing video and force files
        video_file: Video filename
        force_file: Force data filename
    
    Returns:
        tuple: (lag_frames, df_aligned)
    """
    print("\n" + "="*70)
    print(f"LED SYNCING WITH DETECTION SYSTEM - {view}")
    print("="*70)
    
    video_path = os.path.join(parent_path, video_file)
    force_path = os.path.join(parent_path, force_file)
    
    # Create output directory for detection diagnostics
    output_dir = os.path.join(parent_path, "led_detection_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================================================
    # STEP 1: USE DETECTION SYSTEM TO FIND LED AND EXTRACT SIGNAL
    # ========================================================================
    print("\n[STEP 1] Using LED detection system to find LED location...")
    
    try:
        # Process the video to get LED signal
        # This automatically:
        # - Detects LED location using view-specific templates
        # - Extracts Red channel signal
        # - Creates diagnostic images
        # - Saves signal to CSV
        df_video_signal = process_view(
            video_path=video_path,
            view_type=view,
            output_path=output_dir
        )
        
        print(f"[SUCCESS] LED signal extracted: {len(df_video_signal)} frames")
        
    except Exception as e:
        print(f"[ERROR] LED detection system failed: {e}")
        print("[FALLBACK] Using original template matching method...")
        # Fallback to original method
        return new_led_original(self, view, parent_path, video_file, force_file)
    
    # ========================================================================
    # STEP 2: GET TRANSFORMATION MATRIX FOR FORCE PLATE CORNERS
    # ========================================================================
    print("\n[STEP 2] Finding force plate corners in video...")
    
    try:
        M = plate_transformation_matrix(self, view, video_path)
    except Exception as e:
        print(f"[ERROR] Could not find force plate corners: {e}")
        # Use identity matrix as fallback
        M = np.eye(3)
    
    # ========================================================================
    # STEP 3: PROCESS FORCE FILE
    # ========================================================================
    print("\n[STEP 3] Processing force data...")
    
    df_force = process_force_file(M, parent_path, force_file)
    print(f"Force data processed: {len(df_force)} rows")
    
    # ========================================================================
    # STEP 4: ALIGN VIDEO AND FORCE SIGNALS
    # ========================================================================
    print("\n[STEP 4] Aligning video and force signals...")
    
    # The video signal from detection system has 'Video_LED_Signal' column
    # The force signal has 'FP_LED_Signal' column
    
    df_aligned, lag, max_corr, perfect_corr, relative_score = align_data(
        self, df_force, df_video_signal
    )
    
    print(f"\n{'='*70}")
    print("ALIGNMENT RESULTS:")
    print(f"  Lag: {lag} frames")
    print(f"  Correlation: {max_corr:.2f} / {perfect_corr:.2f}")
    print(f"  Relative Score: {relative_score:.4f}")
    print(f"  Aligned data: {len(df_aligned)} rows")
    print("="*70 + "\n")
    
    # ========================================================================
    # STEP 5: SAVE RESULTS
    # ========================================================================
    result_filename = force_file.replace('.txt', '_Results_DetectionSystem.csv')
    result_path = os.path.join(parent_path, result_filename)
    
    df_result = pd.DataFrame([[
        video_file,
        force_file,
        lag,
        max_corr,
        perfect_corr,
        relative_score,
        view
    ]], columns=[
        'Video File',
        'Force File',
        'Video Frame for t_zero force',
        'Correlation Score',
        'Perfect Score',
        'Relative Score',
        'View Type'
    ])
    
    df_result.to_csv(result_path, index=False)
    print(f"[SAVED] Results to: {result_path}")
    
    # Also save diagnostic plot
    _save_alignment_plot(df_aligned, lag, output_dir, view)
    
    return lag, df_aligned


def new_led_original(self, view, parent_path, video_file, force_file):
    """
    Original LED alignment method (fallback if detection system fails).
    This is the existing new_led function.
    """
    from GUI.callbacks.ledSyncing import (
        plate_transformation_matrix,
        process_force_file,
        get_alignment_signal_from_video,
        align_data
    )
    
    video_path = os.path.join(parent_path, video_file)
    
    # Original method: template matching
    M = plate_transformation_matrix(self, view, video_path)
    df_force = process_force_file(M, parent_path, force_file)
    df_video = get_alignment_signal_from_video(self, view, video_path, video_file)
    df_aligned, lag, max_corr, perfect_corr, relative_score = align_data(
        self, df_force, df_video
    )
    
    # Save results
    df_result = pd.DataFrame([[
        video_file, force_file, lag, max_corr, perfect_corr, relative_score
    ]], columns=[
        'Video File', 'Force File', 'Video Frame for t_zero force',
        'Correlation Score', 'Perfect Score', 'Relative Score'
    ])
    
    result_filename = force_file.replace('.txt', '_Results.csv')
    df_result.to_csv(os.path.join(parent_path, result_filename), index=False)
    
    return lag, df_aligned


def _save_alignment_plot(df_aligned, lag, output_dir, view):
    """
    Save a diagnostic plot showing the alignment quality.
    
    Args:
        df_aligned: Aligned DataFrame
        lag: Frame lag
        output_dir: Where to save the plot
        view: View type name
    """
    try:
        # Extract signals
        video_signal = df_aligned['Video_LED_Signal'].to_numpy()
        force_signal = df_aligned['FP_LED_Signal'].to_numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot signals
        frames = np.arange(len(video_signal))
        ax.step(frames, video_signal, where='mid', alpha=0.7, label='Video Signal', linewidth=1.5)
        ax.step(frames, force_signal, where='mid', alpha=0.7, label='Force Signal', linewidth=1.5)
        
        # Add title and labels
        ax.set_title(f"LED Alignment - {view} (Lag = {lag} frames)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Frame Number", fontsize=12)
        ax.set_ylabel("Signal", fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Zoom to middle portion for detail
        n = len(frames)
        if n > 1000:
            ax.set_xlim(n//2 - 500, n//2 + 500)
        
        # Save
        plot_path = os.path.join(output_dir, f"alignment_plot_{view.replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[SAVED] Alignment plot to: {plot_path}")
        
    except Exception as e:
        print(f"[WARNING] Could not save alignment plot: {e}")


def new_led(self, view, parent_path, video_file, force_file, use_detection_system=True):
    """
    Run LED syncing with optional new detection system. (plate swap)
    
    Args:
        view: View type ("Long View", "Top View", "Side1 View", "Side2 View")
        use_detection_system: If True, use new LED detection system with plate swap
    """
    import time
    startTime = time.time()
    
    video_path = os.path.join(parent_path, video_file)
    
    if use_detection_system:
        # Use new detection system
        from GUI.callbacks.led_detection_system import process_view, config_map
        
        print(f"\n[INFO] Using NEW LED detection system for {view}")
        
        # Get config to check plate_swap setting
        config_class = config_map.get(view)
        if config_class is None:
            raise ValueError(f"Unknown view: {view}")
        
        config = config_class()
        should_swap = config.plate_swap
        
        print(f"[INFO] Force plate swap for {view}: {'YES' if should_swap else 'NO'}")
        
        # Process video to get LED signal
        df_video = process_view(video_path, view, parent_path)
        
        # Load and process force data (same as before)
        force_path = os.path.join(parent_path, force_file)
        df_force = pd.read_csv(force_path, header=17, delimiter='\t', encoding='latin1').drop(0)
        df_force = df_force.apply(pd.to_numeric, errors='coerce')
        
        # Rename columns
        force_dict = {
            'abs time (s)': 'Time(s)',
            'Fx': 'FP1_Fx', 'Fy': 'FP1_Fy', 'Fz': 'FP1_Fz', '|Ft|': 'FP1_|F|', 
            'Ax': 'FP1_Ax', 'Ay': 'FP1_Ay',
            'Fx.1': 'FP2_Fx', 'Fy.1': 'FP2_Fy', 'Fz.1': 'FP2_Fz', '|Ft|.1': 'FP2_|F|', 
            'Ax.1': 'FP2_Ax', 'Ay.1': 'FP2_Ay',
            'Fx.2': 'FP3_Fx', 'Fy.2': 'FP3_Fy', 'Fz.2': 'FP3_Fz', '|Ft|.2': 'FP3_|F|', 
            'Ax.2': 'FP3_Ax', 'Ay.2': 'FP3_Ay'
        }
        df_force.rename(columns=force_dict, inplace=True)
        
        # Create LED signal from force data
        df_force['FP_LED_Signal'] = np.sign(df_force['FP3_Fz'])
        
        # Align force and video (same logic as original)
        df_force_subset = df_force.iloc[::10].reset_index(drop=True)
        signal_force = df_force_subset['FP_LED_Signal']
        signal_video = df_video['Video_LED_Signal']
        
        # Z-normalize signals
        video_arr = np.asarray(signal_video, dtype=float)
        force_arr = np.asarray(signal_force, dtype=float)
        
        if np.std(video_arr) > 0:
            video_arr = (video_arr - np.mean(video_arr)) / np.std(video_arr)
        if np.std(force_arr) > 0:
            force_arr = (force_arr - np.mean(force_arr)) / np.std(force_arr)
        
        from scipy import signal
        correlation = signal.correlate(video_arr, force_arr, mode="valid")
        lags = signal.correlation_lags(video_arr.size, force_arr.size, mode="valid")
        lag = lags[np.argmax(correlation)]
        
        print(f"[INFO] Detected lag: {lag} frames")
        
        # Create aligned dataframe
        df_force_subset['FrameNumber'] = list(range(lag, lag + len(df_force_subset)))
        df_aligned = pd.merge(df_force_subset, df_video, on='FrameNumber', how='left')
        
        # **CRITICAL: Apply plate swap if needed**
        if should_swap:
            print(f"\n{'='*60}")
            print(f"[SWAP] Applying force plate swap for {view}")
            print(f"{'='*60}")
            df_aligned = swap_force_plates(df_aligned)
        
        # Save results
        max_corr = float(np.max(correlation))
        perfect_corr = min(len(force_arr), len(video_arr))
        relative_score = max_corr / perfect_corr
        print(f"[INFO] Max Corr: {max_corr:.2f}, Perfect Corr: {perfect_corr:.2f}, Relative Score: {relative_score:.4f}")
        
        df_result = pd.DataFrame([[
            video_file, force_file, lag, max_corr, perfect_corr, relative_score, should_swap
        ]], columns=[
            'Video File', 'Force File', 'Video Frame for t_zero force',
            'Correlation Score', 'Perfect Score', 'Relative Score', 'Plate Swapped'
        ])
        
        df_result_filename = force_file.replace('.txt', '_Results.csv')
        df_result.to_csv(os.path.join(parent_path, df_result_filename), index=False)
        
        print(f"[INFO] LED sync complete. Time: {time.time() - startTime:.2f}s")
        print(f"[INFO] Plate swap applied: {should_swap}")
        
        return lag, df_aligned
    
    else:
        # Use original detection method (no plate swap)
        print(f"\n[INFO] Using ORIGINAL LED detection method")
        from GUI.callbacks.ledSyncing import run_led_syncing
        lag = run_led_syncing(self, parent_path, video_file, force_file)
        # Note: Original method doesn't return df_aligned, so you'd need to construct it
        # For now, raising an error to force use of new system
        raise NotImplementedError("Original method doesn't support df_aligned return")


# For backward compatibility, keep old function name
def run_led_syncing(self, parent_path, video_file, force_file):
    """
    Legacy function for backward compatibility.
    This is the old entry point that returns just the lag value.
    """
    # Determine view from self if available
    view = self.selected_view.get() if hasattr(self, 'selected_view') else "Long View"
    
    lag, _ = new_led(self, view, parent_path, video_file, force_file)
    return lag