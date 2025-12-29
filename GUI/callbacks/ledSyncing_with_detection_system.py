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
    ShortViewLEDConfig,
    LEDDetector
)

# Keep original functions for backward compatibility
from GUI.callbacks.ledSyncing import (
    plate_transformation_matrix,
    process_force_file,
    align_data
)


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
    Main entry point for LED alignment.
    
    Args:
        self: GUI instance
        view: Camera view ("Long View", "Top View", or "Short View")
        parent_path: Directory containing files
        video_file: Video filename
        force_file: Force data filename
        use_detection_system: If True, use new detection system; if False, use original
    
    Returns:
        tuple: (lag_frames, df_aligned)
    """
    if use_detection_system:
        try:
            return new_led_with_detection_system(self, view, parent_path, video_file, force_file)
        except Exception as e:
            print(f"[ERROR] Detection system failed: {e}")
            print("[FALLBACK] Using original method...")
            return new_led_original(self, view, parent_path, video_file, force_file)
    else:
        return new_led_original(self, view, parent_path, video_file, force_file)


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