"""
Utility for finding force boundaries in aligned data.
This determines the subset of frames to process based on force threshold.
"""
import pandas as pd
import numpy as np


def find_force_boundaries(df_aligned, threshold=50, padding_frames=10):
    """
    Find the start and end frames where force exceeds threshold.
    
    Args:
        df_aligned: DataFrame with aligned force and video data
        threshold: Minimum force in Newtons (default 50)
        padding_frames: Extra frames to include before/after (default 10)
    
    Returns:
        tuple: (boundary_start_frame, boundary_end_frame)
    """
    print(f"\n========== Finding Force Boundaries (threshold={threshold}N) ==========")
    
    # Calculate max force across both plates
    if 'FP1_|F|' in df_aligned.columns and 'FP2_|F|' in df_aligned.columns:
        df_aligned['MaxForce'] = df_aligned[['FP1_|F|', 'FP2_|F|']].max(axis=1)
    else:
        # Try alternative column names
        force_cols = [col for col in df_aligned.columns if '|F|' in col or 'Fz' in col]
        if len(force_cols) >= 2:
            df_aligned['MaxForce'] = df_aligned[force_cols[:2]].max(axis=1)
        else:
            raise ValueError("Could not find force magnitude columns in df_aligned")
    
    # Find frames where force >= threshold
    force_frames = df_aligned[df_aligned['MaxForce'] >= threshold].copy()
    
    if len(force_frames) == 0:
        print(f"[WARNING] No frames found with force >= {threshold}N")
        print(f"Max force in data: {df_aligned['MaxForce'].max():.2f}N")
        # Return full range as fallback
        return int(df_aligned['FrameNumber'].min()), int(df_aligned['FrameNumber'].max())
    
    # Get boundary frames
    start_frame = int(force_frames['FrameNumber'].min())
    end_frame = int(force_frames['FrameNumber'].max())
    
    # Apply padding (but keep within valid range)
    min_frame = int(df_aligned['FrameNumber'].min())
    max_frame = int(df_aligned['FrameNumber'].max())
    
    boundary_start = max(min_frame, start_frame - padding_frames)
    boundary_end = min(max_frame, end_frame + padding_frames)
    
    # Calculate statistics
    total_frames = len(df_aligned)
    trimmed_frames = boundary_end - boundary_start + 1
    percent_kept = (trimmed_frames / total_frames) * 100
    
    print(f"Total frames in df_aligned: {total_frames}")
    print(f"Frames with force >= {threshold}N: {len(force_frames)}")
    print(f"Boundary frames: {boundary_start} to {boundary_end}")
    print(f"Trimmed subset: {trimmed_frames} frames ({percent_kept:.1f}% of total)")
    print(f"Max force in subset: {force_frames['MaxForce'].max():.2f}N")
    print("=" * 70 + "\n")
    
    return boundary_start, boundary_end


def get_trimmed_subset(df_aligned, boundary_start, boundary_end):
    """
    Extract the trimmed subset from df_aligned.
    
    Args:
        df_aligned: Full aligned DataFrame
        boundary_start: Start frame number
        boundary_end: End frame number
    
    Returns:
        DataFrame: Trimmed subset
    """
    # print("[In Trimming Method] Columns in df_aligned:", list(df_aligned.columns))
    df_trimmed = df_aligned[
        (df_aligned['FrameNumber'] >= boundary_start) & 
        (df_aligned['FrameNumber'] <= boundary_end)
    ].copy()
    
    return df_trimmed