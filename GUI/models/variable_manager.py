class StateManager:
    def __init__(self):
        # Global flags
        self.force_loaded = False
        self.video_loaded = False
        self.vector_overlay_enabled = False
        self.com_enabled = False
        
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
        self.step_size      = 10    # step size unit: rows/frame
        self.zoom_pos       = 0     # canvas 2: force data offset -step size<zoom_pos<+step size
        self.force_align    = None  # Intialize force align value and video align value

        # video
        self.rot = 0 # rotated direction
        self.video_align = None
