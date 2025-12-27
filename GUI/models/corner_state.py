"""
Model for storing force plate corner coordinates detected from video.
These corners are used for perspective transformation when overlaying force vectors.
"""

class CornerState:
    def __init__(self):
        # Store 8 corner points: 4 for each force plate
        # Format: [[x, y], [x, y], ...] for 8 points total
        self.corners = []
        self.plate1_corners = []  # First 4 points
        self.plate2_corners = []  # Last 4 points
        
    def set_corners(self, corners_list):
        """
        Set all 8 corner points at once.
        
        Args:
            corners_list: List of 8 [x, y] coordinate pairs
        """
        if len(corners_list) != 8:
            raise ValueError(f"Expected 8 corners, got {len(corners_list)}")
        
        self.corners = corners_list
        self.plate1_corners = corners_list[0:4]
        self.plate2_corners = corners_list[4:8]
        
    def get_all_corners(self):
        """Return all 8 corners as a list."""
        return self.corners
    
    def get_plate1_corners(self):
        """Return corners for force plate 1 (first 4 points)."""
        return self.plate1_corners
    
    def get_plate2_corners(self):
        """Return corners for force plate 2 (last 4 points)."""
        return self.plate2_corners
    
    def has_corners(self):
        """Check if corners have been set."""
        return len(self.corners) == 8
    
    def clear(self):
        """Reset all corner data."""
        self.corners = []
        self.plate1_corners = []
        self.plate2_corners = []