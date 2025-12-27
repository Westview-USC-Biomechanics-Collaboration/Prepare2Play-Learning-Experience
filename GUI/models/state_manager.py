from GUI.models.corner_state import CornerState

class StateManager:
    def __init__(self):
        # Global flags
        self.force_loaded = False
        self.video_loaded = False
        self.vector_overlay_enabled = False
        self.com_enabled = False
        self.corner_state = CornerState()

        # Global Flags - old
        # self.force_data_flag = False
        # self.video_data_flag = False
        # self.vector_overlay_flag = False
        # self.COM_flag = False

        # Global frame/location base on slider
        self.loc = 0

        # State variables
        # force data
        self.force_start    = None  # This variable store the time in raw force data which user choose to align
        self.force_frame    = None  # total number of frames could be represented by force data ->calculation: TotalRows/stepsize
        self.step_size      = 10    # step siize unit: rows/frame
        self.zoom_pos       = 0     # canvas 2: force data offset -step size<zoom_pos<+step size
        self.force_align    = None  # Intialize force align value and video align value
        self.df_aligned = None

        # video
        self.rot = 0 # rotated direction
        self.video_align = None

        # Slider bounds management
        self.slider_min = 0
        self.slider_max = 100  # Default, updated when video loads
        self.is_using_trimmed = False  # Track if we're viewing trimmed content
        
    def get_safe_slider_value(self, requested_value):
        """
        Ensure slider value is within valid bounds.
        
        Args:
            requested_value: The frame number requested
            
        Returns:
            int: Clamped value within [slider_min, slider_max]
        """
        return max(self.slider_min, min(requested_value, self.slider_max))
    
    def update_slider_bounds(self, min_val, max_val):
        """
        Update the valid range for the slider.
        
        Args:
            min_val: Minimum frame number
            max_val: Maximum frame number
        """
        self.slider_min = int(min_val)
        self.slider_max = int(max_val)
        print(f"[StateManager] Updated slider bounds: {self.slider_min} to {self.slider_max}")
    
    def get_slider_bounds(self):
        """Get current slider bounds as tuple."""
        return (self.slider_min, self.slider_max)

