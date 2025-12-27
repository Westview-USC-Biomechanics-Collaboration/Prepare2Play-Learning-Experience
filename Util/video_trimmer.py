"""
Utility for creating a trimmed copy of video based on force data.
Removes dead time before and after significant forces are detected.
"""

import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Note: Force data column names vary depending on which processing step you're at:
# - Raw file: 'Fx', 'Fy', 'Fz', '|Ft|', 'Fx.1', 'Fy.1', etc.
# - After FileFormatter: 'Fx1', 'Fy1', 'Fz1', '|Ft1|', 'Fx2', etc.
# - After vectoroverlay rename: 'FP1_Fx', 'FP1_Fy', 'FP1_|F|', etc.
# This class handles all three cases automatically.


class VideoTrimmer:
    def __init__(self, video_path, force_data, force_threshold=50.0, step_size=10):
        """
        Initialize the video trimmer.
        
        Args:
            video_path: Path to the video file
            force_data: DataFrame with force data (must have 'FP1_|F|', 'FP2_|F|' columns)
            force_threshold: Minimum force magnitude to consider significant (Newtons)
            step_size: Downsampling factor for force data (e.g., 10 means 10 force samples per video frame)
        """
        self.video_path = video_path
        self.force_data = force_data
        self.force_threshold = force_threshold
        self.step_size = step_size
        
        # Video properties (will be set when video is opened)
        self.fps = None
        self.frame_width = None
        self.frame_height = None
        self.total_frames = None
        
        # Trim boundaries (in frames)
        self.trim_start_frame = None
        self.trim_end_frame = None
        
    def _get_video_properties(self):
        """Extract video properties."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()
        
        print(f"[VideoTrimmer] Video properties:")
        print(f"  FPS: {self.fps}")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        print(f"  Total frames: {self.total_frames}")
        
    def calculate_trim_boundaries(self):
        """
        Calculate which frames to keep based on force data.
        Works with both raw force data and df_aligned format.
        
        Returns:
            tuple: (start_frame, end_frame) - inclusive boundaries
        """
        self._get_video_properties()
        
        # Print available columns for debugging
        print(f"[VideoTrimmer] Available force data columns: {list(self.force_data.columns)}")
        
        # Check if this is df_aligned (has FrameNumber column)
        is_aligned = 'FrameNumber' in self.force_data.columns
        
        if is_aligned:
            print("[VideoTrimmer] Processing df_aligned format")
        else:
            print("[VideoTrimmer] Processing raw force data format")
        
        # Try to find the magnitude force columns with different possible names
        force_mag_cols = []
        possible_names = [
            ['FP1_|F|', 'FP2_|F|'],      # After rename in vectoroverlay_GUI
            ['|Ft1|', '|Ft2|'],          # Original names from file
            ['|Ft|', '|Ft|.1'],          # Raw file format
        ]
        
        for name_pair in possible_names:
            if all(col in self.force_data.columns for col in name_pair):
                force_mag_cols = name_pair
                print(f"[VideoTrimmer] Using force magnitude columns: {force_mag_cols}")
                break
        
        # If no magnitude columns found, calculate from Fx, Fy, Fz
        if not force_mag_cols:
            print("[VideoTrimmer] No magnitude columns found, calculating from components...")
            
            # Try different naming conventions
            if 'FP1_Fx' in self.force_data.columns:
                # After rename
                self.force_data['FP1_|F|'] = np.sqrt(
                    self.force_data['FP1_Fx']**2 + 
                    self.force_data['FP1_Fy']**2 + 
                    self.force_data['FP1_Fz']**2
                )
                self.force_data['FP2_|F|'] = np.sqrt(
                    self.force_data['FP2_Fx']**2 + 
                    self.force_data['FP2_Fy']**2 + 
                    self.force_data['FP2_Fz']**2
                )
                force_mag_cols = ['FP1_|F|', 'FP2_|F|']
            elif 'Fx1' in self.force_data.columns:
                # After file reader rename
                self.force_data['|Ft1|'] = np.sqrt(
                    self.force_data['Fx1']**2 + 
                    self.force_data['Fy1']**2 + 
                    self.force_data['Fz1']**2
                )
                self.force_data['|Ft2|'] = np.sqrt(
                    self.force_data['Fx2']**2 + 
                    self.force_data['Fy2']**2 + 
                    self.force_data['Fz2']**2
                )
                force_mag_cols = ['|Ft1|', '|Ft2|']
            else:
                raise ValueError(
                    f"Cannot find force columns. Available columns: {list(self.force_data.columns)}"
                )
            
            print(f"[VideoTrimmer] Calculated magnitude columns: {force_mag_cols}")
        
        # Calculate max force across both plates
        if 'MaxForce' not in self.force_data.columns:
            self.force_data['MaxForce'] = self.force_data[force_mag_cols].max(axis=1)
        
        # Find rows where force exceeds threshold
        significant_force_mask = self.force_data['MaxForce'] >= self.force_threshold
        significant_indices = self.force_data[significant_force_mask].index
        
        if len(significant_indices) == 0:
            print(f"[WARNING] No forces above {self.force_threshold}N detected!")
            return 0, self.total_frames - 1
        
        # Get first and last significant force rows
        first_force_row = significant_indices[0]
        last_force_row = significant_indices[-1]
        
        print(f"[VideoTrimmer] Significant force detected:")
        print(f"  First row index: {first_force_row}")
        print(f"  Last row index: {last_force_row}")
        
        # Convert to video frame numbers
        if is_aligned:
            # df_aligned already has FrameNumber column
            self.trim_start_frame = int(self.force_data.loc[first_force_row, 'FrameNumber'])
            self.trim_end_frame = int(self.force_data.loc[last_force_row, 'FrameNumber'])
        else:
            # Raw force data: convert using step_size
            self.trim_start_frame = max(0, int(first_force_row / self.step_size))
            self.trim_end_frame = min(self.total_frames - 1, int(last_force_row / self.step_size))
        
        print(f"[VideoTrimmer] Calculated trim boundaries:")
        print(f"  Start frame: {self.trim_start_frame}")
        print(f"  End frame: {self.trim_end_frame}")
        print(f"  Trimmed duration: {(self.trim_end_frame - self.trim_start_frame) / self.fps:.2f} seconds")
        
        return self.trim_start_frame, self.trim_end_frame
    
    def create_trimmed_video(self, output_path=None):
        """
        Create a new trimmed video file.
        
        Args:
            output_path: Path for output video. If None, adds "_trimmed" suffix to original name.
            
        Returns:
            str: Path to the created trimmed video
        """
        if self.trim_start_frame is None or self.trim_end_frame is None:
            self.calculate_trim_boundaries()
        
        # Generate output path if not provided
        if output_path is None:
            video_path_obj = Path(self.video_path)
            output_path = str(video_path_obj.parent / f"{video_path_obj.stem}_trimmed{video_path_obj.suffix}")
        
        # Open input video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        # Set up output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        print(f"[VideoTrimmer] Creating trimmed video: {output_path}")
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.trim_start_frame)
        
        # Copy frames from start to end
        frame_count = 0
        for frame_idx in range(self.trim_start_frame, self.trim_end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"[WARNING] Could not read frame {frame_idx}")
                break
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames...")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"[VideoTrimmer] Successfully created trimmed video with {frame_count} frames")
        
        return output_path
    
    def get_trimmed_frame_mapping(self):
        """
        Get a mapping from trimmed video frames to original video frames.
        
        Returns:
            dict: {trimmed_frame_index: original_frame_index}
        """
        if self.trim_start_frame is None:
            self.calculate_trim_boundaries()
        
        return {
            i: self.trim_start_frame + i 
            for i in range(self.trim_end_frame - self.trim_start_frame + 1)
        }


# Usage example
if __name__ == "__main__":
    # Example usage
    video_path = "path/to/video.mp4"
    force_file = "path/to/force_data.txt"
    
    # Load force data (adjust based on your file format)
    force_data = pd.read_csv(force_file, skiprows=19)
    
    # Create trimmer
    trimmer = VideoTrimmer(
        video_path=video_path,
        force_data=force_data,
        force_threshold=50.0,
        step_size=10
    )
    
    # Create trimmed video
    trimmed_path = trimmer.create_trimmed_video()
    print(f"Trimmed video saved to: {trimmed_path}")