"""
Updated StateManager with boundary frame storage.
This ensures all views can access the trimmed subset boundaries.
"""

class StateManager:
    def __init__(self):
        # Global flags
        self.force_loaded = False
        self.video_loaded = False
        self.vector_overlay_enabled = False
        self.com_enabled = False
        
        # Global frame/location based on slider
        self.loc = 0

        # State variables - force data
        self.force_start = None      # Time in raw force data for alignment
        self.force_frame = None      # Total frames representable by force data
        self.step_size = 10          # Rows per frame
        self.zoom_pos = 0            # Canvas 2: force data offset
        self.force_align = None      # Force alignment frame
        self.df_aligned = None       # Aligned force and video data
        self.df_trimmed = None      # Trimmed force data subset (uses renamed version of columns - e.g. FP1_fy or something like this)
        

        # State variables - video
        self.rot = 0                 # Rotation direction
        self.video_align = None      # Video alignment frame
        
        # NEW: Boundary frame storage for trimmed processing
        self.boundary_start = None   # First frame with significant force (≥50N)
        self.boundary_end = None     # Last frame with significant force
        self.force_threshold = 50    # Threshold used for boundary detection (N)
        self.boundary_padding = 10   # Extra frames before/after boundaries
        
        # NEW: Processing metadata
        self.processing_metadata = {
            'lag': None,                    # Frame lag from LED alignment
            'total_frames_processed': None, # Number of frames in trimmed subset
            'max_force_in_subset': None,    # Maximum force in trimmed data
            'com_csv_path': None,           # Path to COM data CSV
            'vector_output_path': None      # Path to vector overlay output video
        }
    
    def set_boundaries(self, start_frame, end_frame, threshold=50, padding=10):
        """
        Store boundary frames for trimmed processing.
        
        Args:
            start_frame: First frame to process
            end_frame: Last frame to process
            threshold: Force threshold used (for reference)
            padding: Padding applied (for reference)
        """
        self.boundary_start = start_frame
        self.boundary_end = end_frame
        self.force_threshold = threshold
        self.boundary_padding = padding
        
        print(f"[STATE] Boundaries set: {start_frame} to {end_frame}")
        print(f"[STATE] Threshold: {threshold}N, Padding: {padding} frames")
    
    def get_boundaries(self):
        """
        Get the current boundary frames.
        
        Returns:
            tuple: (boundary_start, boundary_end) or (None, None) if not set
        """
        return self.boundary_start, self.boundary_end
    
    def is_frame_in_boundaries(self, frame_number):
        """
        Check if a frame number is within the processing boundaries.
        
        Args:
            frame_number: Frame number to check
        
        Returns:
            bool: True if frame is within boundaries, False otherwise
        """
        if self.boundary_start is None or self.boundary_end is None:
            return True  # No boundaries set, include all frames
        
        return self.boundary_start <= frame_number <= self.boundary_end
    
    def get_trimmed_frame_count(self):
        """
        Get the number of frames in the trimmed subset.
        
        Returns:
            int: Number of frames, or None if boundaries not set
        """
        if self.boundary_start is None or self.boundary_end is None:
            return None
        
        return self.boundary_end - self.boundary_start + 1
    
    def update_processing_metadata(self, **kwargs):
        """
        Update processing metadata with any key-value pairs.
        
        Example:
            state.update_processing_metadata(
                lag=42,
                total_frames_processed=700,
                com_csv_path='pose_landmarks.csv'
            )
        """
        for key, value in kwargs.items():
            if key in self.processing_metadata:
                self.processing_metadata[key] = value
                print(f"[STATE] Metadata updated: {key} = {value}")
            else:
                print(f"[STATE WARNING] Unknown metadata key: {key}")
    
    def get_processing_summary(self):
        """
        Get a formatted summary of the current processing state.
        
        Returns:
            str: Multi-line summary string
        """
        summary = []
        summary.append("=" * 50)
        summary.append("PROCESSING STATE SUMMARY")
        summary.append("=" * 50)
        
        # Flags
        summary.append(f"Force Loaded: {self.force_loaded}")
        summary.append(f"Video Loaded: {self.video_loaded}")
        summary.append(f"Vector Overlay Enabled: {self.vector_overlay_enabled}")
        summary.append(f"COM Enabled: {self.com_enabled}")
        summary.append("")
        
        # Boundaries
        if self.boundary_start is not None and self.boundary_end is not None:
            summary.append(f"Boundaries: {self.boundary_start} to {self.boundary_end}")
            summary.append(f"Trimmed Frames: {self.get_trimmed_frame_count()}")
            summary.append(f"Force Threshold: {self.force_threshold}N")
            summary.append(f"Padding: {self.boundary_padding} frames")
        else:
            summary.append("Boundaries: Not set")
        summary.append("")
        
        # Metadata
        summary.append("Processing Metadata:")
        for key, value in self.processing_metadata.items():
            summary.append(f"  {key}: {value}")
        
        summary.append("=" * 50)
        return "\n".join(summary)
    
    def reset_boundaries(self):
        """Reset boundary information (useful for reprocessing)."""
        self.boundary_start = None
        self.boundary_end = None
        print("[STATE] Boundaries reset")
    
    def reset_all(self):
        """Reset all state (useful for loading new data)."""
        self.__init__()
        print("[STATE] All state reset")
###---------------------OLD STATE MANAGER---------------------###
# class StateManager:
#     def __init__(self):
#         # Global flags
#         self.force_loaded = False
#         self.video_loaded = False
#         self.vector_overlay_enabled = False
#         self.com_enabled = False
        
#         # Global Flags - old
#         # self.force_data_flag = False
#         # self.video_data_flag = False
#         # self.vector_overlay_flag = False
#         # self.COM_flag = False

#         # Global frame/location base on slider
#         self.loc = 0

#         # State variables
#         # force data
#         self.force_start    = None  # This variable store the time in raw force data which user choose to align
#         self.force_frame    = None  # total number of frames could be represented by force data ->calculation: TotalRows/stepsize
#         self.step_size      = 10    # step siize unit: rows/frame
#         self.zoom_pos       = 0     # canvas 2: force data offset -step size<zoom_pos<+step size
#         self.force_align    = None  # Intialize force align value and video align value
#         self.df_aligned = None

#         # video
#         self.rot = 0 # rotated direction
#         self.video_align = None
