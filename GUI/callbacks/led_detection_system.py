"""
LED Detection and Alignment System for Multiple Camera Views

This module provides robust LED detection for Long View (Front), Top View, and Side Views.
Each view has customized templates and crop regions to handle variations in LED 
appearance and position.

Key Features:
- Automatic Side View classification (Side1 vs Side2) based on LED position
- Support for different resolutions (1920x1080 for Long View, 3840x2160 for GoPro views)
- Configurable force plate swap for each view to maintain consistent orientation

Author: Modified from original USC Biomechanics code
Date: December 2025
"""

import cv2
import os
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class LEDConfig:
    """
    Configuration for LED detection in a specific camera view.
    
    All measurements are in pixels unless otherwise noted.
    Crop regions define where to search for the LED in the full frame.
    
    Attributes:
        view_name: Name of the camera view (e.g., "Long View", "Top View")
        frame_width: Full frame width in pixels
        frame_height: Full frame height in pixels
        led_crop_x0, led_crop_x1: Horizontal bounds of LED search region
        led_crop_y0, led_crop_y1: Vertical bounds of LED search region
        led_template: Template image for LED detection
        template_center_offset_x, template_center_offset_y: Offsets from template corner to LED center
        signal_delta: Radius around LED center for signal averaging
        num_frames_to_check: Number of frames to sample for LED location
        plate_swap: If True, swap FP1 and FP2 data after alignment to maintain consistent orientation
    """
    view_name: str
    
    # Frame dimensions (varies by camera type)
    frame_width: int = 1920
    frame_height: int = 1080
    
    # Crop region for LED search
    led_crop_x0: int = 0
    led_crop_x1: int = 0
    led_crop_y0: int = 0
    led_crop_y1: int = 0
    
    # Template for matching
    led_template: Optional[np.ndarray] = None
    
    # Offset from template top-left corner to LED center (in pixels)
    template_center_offset_x: int = 0
    template_center_offset_y: int = 0
    
    # Region around detected center to average for signal extraction
    signal_delta: int = 3
    
    # Number of frames to sample for robust location detection
    num_frames_to_check: int = 11
    
    # Force plate orientation flag
    plate_swap: bool = False
    
    def __post_init__(self):
        """Create the LED template after initialization"""
        if self.led_template is None:
            self.led_template = self.create_led_template()
    
    def create_led_template(self) -> np.ndarray:
        """
        Create LED template for this view. Override in subclasses.
        
        Returns:
            np.ndarray: Template image for LED detection
        """
        raise NotImplementedError("Subclass must implement create_led_template")
    
    def get_crop_region(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract the crop region from a frame.
        
        Args:
            frame: Full video frame
            
        Returns:
            np.ndarray: Cropped region where LED is expected
        """
        return frame[self.led_crop_y0:self.led_crop_y1, 
                    self.led_crop_x0:self.led_crop_x1, :]
    
    def process_crop_for_matching(self, crop: np.ndarray) -> np.ndarray:
        """
        Process crop for template matching using Blue-Green subtraction.
        
        This method:
        1. Extracts Blue and Green channels
        2. Subtracts Green from Blue (highlights Blue LED, darkens surroundings)
        3. Applies blur to reduce noise
        
        Args:
            crop: Cropped region of frame
            
        Returns:
            np.ndarray: Processed grayscale image ready for template matching
        """
        blue_channel = crop[:, :, 0]
        green_channel = crop[:, :, 1]
        
        # Subtract green from blue - LED appears bright, background dark
        blue_minus_green = cv2.subtract(blue_channel, green_channel)
        
        # Apply blur to reduce noise and make template matching more robust
        processed = cv2.blur(blue_minus_green, (10, 10))
        
        return processed


class LongViewLEDConfig(LEDConfig):
    """
    Configuration for Long View (Front View) camera.
    
    Resolution: 1920x1080 (main camera)
    Force plate orientation: Left = FP1, Right = FP2 (standard, no swap needed)
    LED location: Bottom-center of frame
    """
    def __init__(self):
        super().__init__(
            view_name="Long View",
            frame_width=1920,
            frame_height=1080,
            led_crop_x0=850,
            led_crop_x1=1050,
            led_crop_y0=950,
            led_crop_y1=1080,
            template_center_offset_x=45,
            template_center_offset_y=47,
            plate_swap=False  # Long view: left=FP1, right=FP2 (standard)
        )
    
    def create_led_template(self) -> np.ndarray:
        """
        Create template for Long View LED.
        
        Template structure (71h x 91w pixels):
        - Black background (value 0)
        - Bright rectangle for LED block (value 200)
        - Dark center where actual LEDs are (value 10)
        
        The LED block is positioned toward the bottom of the template to handle
        cases where it's near the edge of the frame.
        
        Returns:
            np.ndarray: LED template image
        """
        template = np.zeros((71, 91), dtype=np.uint8)
        
        # Main bright rectangle (outer LED block)
        cv2.rectangle(template, (20, 27), (71, 68), 200, -1)
        
        # Dark center (where actual LEDs are - brighter in green channel)
        cv2.rectangle(template, (42, 30), (49, 45), 10, -1)
        
        # Blur to match processed images
        template = cv2.blur(template, (5, 5))
        
        return template


class TopViewLEDConfig(LEDConfig):
    """
    Configuration for Top View camera (GoPro).
    
    Resolution: 3840x2160 (GoPro 4K)
    Force plate orientation: When LED is at top-center:
    - Physical left (in video) = FP2
    - Physical right (in video) = FP1
    So we SWAP to match Long View convention
    LED location: Top-center of frame (when camera looks down)
    """
    def __init__(self):
        super().__init__(
            view_name="Top View",
            frame_width=3840,
            frame_height=2160,
            # Scale crop region for 4K resolution (2x the 1080p values)
            led_crop_x0=1400,
            led_crop_x1=2900,
            led_crop_y0=100,
            led_crop_y1=2100,
            template_center_offset_x=45,
            template_center_offset_y=44,
            plate_swap=True  # Top view: swap FP1/FP2 to match Long View orientation
        )
    
    def create_led_template(self) -> np.ndarray:
        """
        Create template for Top View LED.
        
        Similar structure to Long View but with slightly different dimensions
        to account for viewing angle differences.
        
        Returns:
            np.ndarray: LED template image
        """
        template = np.zeros((77, 91), dtype=np.uint8)
        
        # Main bright rectangle
        cv2.rectangle(template, (20, 20), (71, 57), 200, -1)
        
        # Dark center where LEDs are
        cv2.rectangle(template, (43, 27), (47, 32), 10, -1)
        
        # Blur to match processed images
        template = cv2.blur(template, (5, 5))
        
        return template


class SideViewLEDConfig(LEDConfig):
    """
    Base configuration for Side View cameras (GoPro).
    
    Resolution: 3840x2160 (GoPro 4K)
    
    NOTE: This is a base class. Use auto_detect_side_view() to automatically
    determine if this is Side1 or Side2 based on LED position, then create
    the appropriate Side1ViewLEDConfig or Side2ViewLEDConfig.
    
    Side1: LED on left, Near plate = FP2, Far plate = FP1 (SWAP needed)
    Side2: LED on right, Near plate = FP1, Far plate = FP2 (NO swap)
    """
    def __init__(self, is_side1: bool):
        """
        Initialize Side View configuration.
        
        Args:
            is_side1: True if LED is on left side (Side1), False if on right (Side2)
        """
        if is_side1:
            # Side1: LED on left
            view_name = "Side1 View"
            led_crop_x0 = 200
            led_crop_x1 = 800
            plate_swap = True  # Side1: near=FP2, far=FP1, so swap
        else:
            # Side2: LED on right
            view_name = "Side2 View"
            led_crop_x0 = 3040
            led_crop_x1 = 3640
            plate_swap = False  # Side2: near=FP1, far=FP2 (standard)
        
        super().__init__(
            view_name=view_name,
            frame_width=3840,
            frame_height=2160,
            led_crop_x0=led_crop_x0,
            led_crop_x1=led_crop_x1,
            led_crop_y0=800,
            led_crop_y1=1600,
            template_center_offset_x=45,
            template_center_offset_y=47,
            plate_swap=plate_swap
        )
    
    def create_led_template(self) -> np.ndarray:
        """
        Create template for Side View LED.
        
        Uses similar structure to Long View template.
        
        Returns:
            np.ndarray: LED template image
        """
        template = np.zeros((71, 91), dtype=np.uint8)
        
        # Main bright rectangle
        cv2.rectangle(template, (20, 27), (71, 68), 200, -1)
        
        # Dark center where LEDs are
        cv2.rectangle(template, (42, 30), (49, 45), 10, -1)
        
        # Blur to match processed images
        template = cv2.blur(template, (5, 5))
        
        return template


def auto_detect_side_view(video_path: str) -> LEDConfig:
    """
    Automatically detect if a side view video is Side1 or Side2 based on LED position.
    
    Process:
    1. Open video and read first frame
    2. Search for LED in left third of frame
    3. Search for LED in right third of frame
    4. Compare signal strengths to determine which side has the LED
    5. Return appropriate SideViewLEDConfig
    
    Args:
        video_path: Path to side view video file
        
    Returns:
        LEDConfig: Either Side1ViewLEDConfig or Side2ViewLEDConfig
        
    Raises:
        ValueError: If LED cannot be detected in either side
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Cannot read first frame from: {video_path}")
    
    h, w = frame.shape[:2]
    
    # Define search regions for left and right sides
    # Left third of frame
    left_crop = frame[h//3:2*h//3, 0:w//3, :]
    # Right third of frame
    right_crop = frame[h//3:2*h//3, 2*w//3:w, :]
    
    # Process both crops using Blue-Green subtraction
    def get_led_signal(crop):
        """Calculate average LED signal strength in crop"""
        blue = crop[:, :, 0].astype(float)
        green = crop[:, :, 1].astype(float)
        blue_minus_green = np.clip(blue - green, 0, 255)
        # Return 95th percentile (robust to outliers)
        return np.percentile(blue_minus_green, 95)
    
    left_signal = get_led_signal(left_crop)
    right_signal = get_led_signal(right_crop)
    
    print(f"\nSide View Auto-Detection:")
    print(f"  Left side signal strength: {left_signal:.1f}")
    print(f"  Right side signal strength: {right_signal:.1f}")
    
    # Threshold: LED side should have significantly higher signal
    # Require at least 50 units difference to be confident
    if left_signal > right_signal + 50:
        print(f"  → Detected as Side1 View (LED on left)")
        return SideViewLEDConfig(is_side1=True)
    elif right_signal > left_signal + 50:
        print(f"  → Detected as Side2 View (LED on right)")
        return SideViewLEDConfig(is_side1=False)
    else:
        # If signals are too close, default to Side1 and warn user
        print(f"  ⚠ WARNING: Cannot confidently determine side (signals too similar)")
        print(f"  → Defaulting to Side1 View")
        return SideViewLEDConfig(is_side1=True)


class LEDDetector:
    """
    Detects LED location in video and extracts alignment signal.
    
    This class:
    1. Finds LED location by sampling multiple frames
    2. Extracts Red channel signal from LED for alignment
    3. Creates clean binary signal for correlation with force data
    4. Generates diagnostic images showing detection results
    """
    
    def __init__(self, config: LEDConfig):
        """
        Initialize detector with view-specific configuration.
        
        Args:
            config: LEDConfig subclass with view-specific parameters
        """
        self.config = config
    
    def find_led_location(
        self, 
        video_path: str, 
        output_path: str
    ) -> Tuple[int, int]:
        """
        Find LED center location by sampling multiple frames.
        
        Strategy:
        1. Sample frames evenly distributed across video
        2. Find LED in each frame using template matching
        3. Use median X and median Y as final location (robust to outliers)
        4. Save diagnostic images showing detection in each frame
        
        Args:
            video_path: Path to video file
            output_path: Path to save diagnostic images
            
        Returns:
            (center_x, center_y): LED center in full frame coordinates
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Storage for detected locations
        locations_df = pd.DataFrame(
            columns=['FrameNumber', 'CenterX_Crop', 'CenterY_Crop', 
                    'CenterX_Full', 'CenterY_Full']
        )
        
        # Storage for diagnostic images
        diagnostic_images = []
        
        # Sample frames evenly across video
        frame_indices = np.linspace(
            0, total_frames - 1, 
            self.config.num_frames_to_check, 
            dtype=int
        )
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Warning: Could not read frame {frame_idx}")
                continue
            
            # Extract crop region
            crop = self.config.get_crop_region(frame)
            
            # Process for template matching
            processed = self.config.process_crop_for_matching(crop)
            
            # Find LED using template matching
            result = cv2.matchTemplate(
                processed, 
                self.config.led_template, 
                cv2.TM_SQDIFF  # Minimize squared difference
            )
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Calculate center in crop coordinates
            center_x_crop = min_loc[0] + self.config.template_center_offset_x
            center_y_crop = min_loc[1] + self.config.template_center_offset_y
            
            # Convert to full frame coordinates
            center_x_full = center_x_crop + self.config.led_crop_x0
            center_y_full = center_y_crop + self.config.led_crop_y0
            
            # Store location
            locations_df = pd.concat([
                locations_df,
                pd.DataFrame([{
                    'FrameNumber': frame_idx,
                    'CenterX_Crop': center_x_crop,
                    'CenterY_Crop': center_y_crop,
                    'CenterX_Full': center_x_full,
                    'CenterY_Full': center_y_full
                }])
            ], ignore_index=True)
            
            # Create diagnostic image for this frame
            diagnostic = self._create_diagnostic_image(
                crop, processed, center_x_crop, center_y_crop, frame_idx
            )
            diagnostic_images.append(diagnostic)
        
        cap.release()
        
        # Calculate median location (robust to outliers)
        median_x = int(locations_df['CenterX_Full'].median())
        median_y = int(locations_df['CenterY_Full'].median())
        
        # Calculate span to check consistency
        span_x = int(locations_df['CenterX_Full'].max() - locations_df['CenterX_Full'].min())
        span_y = int(locations_df['CenterY_Full'].max() - locations_df['CenterY_Full'].min())
        
        print(f"\n{self.config.view_name} LED Detection Results:")
        print(f"  Median location: ({median_x}, {median_y})")
        print(f"  X span: {span_x} pixels, Y span: {span_y} pixels")
        print(f"  Force plate swap: {'YES (FP1 ↔ FP2)' if self.config.plate_swap else 'NO (standard)'}")
        
        if span_x > 40 or span_y > 40:  # Adjusted threshold for 4K resolution
            print(f"  ⚠ WARNING: Large variation in detected location!")
            print(f"  This may indicate detection issues. Check diagnostic images.")
        
        # Save results
        self._save_detection_results(
            locations_df, diagnostic_images, output_path, 
            median_x, median_y, span_x, span_y
        )
        
        return median_x, median_y
    
    def _create_diagnostic_image(
        self, 
        crop: np.ndarray, 
        processed: np.ndarray,
        center_x: int, 
        center_y: int, 
        frame_num: int
    ) -> np.ndarray:
        """
        Create diagnostic image showing detection for one frame.
        
        Args:
            crop: Original cropped region
            processed: Processed crop used for detection
            center_x: Detected LED center X in crop coordinates
            center_y: Detected LED center Y in crop coordinates
            frame_num: Frame number
            
        Returns:
            np.ndarray: Composite diagnostic image
        """
        # Annotate crop with detected center
        crop_annotated = crop.copy()
        cv2.line(crop_annotated, (center_x, center_y - 20), 
                (center_x, center_y + 20), (0, 0, 255), 2)
        cv2.line(crop_annotated, (center_x - 20, center_y), 
                (center_x + 20, center_y), (0, 0, 255), 2)
        
        # Create info panel
        info = np.zeros((150, crop.shape[1], 3), dtype=np.uint8)
        cv2.putText(info, f"Frame: {frame_num}", (10, 35), 
                   cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(info, f"Center (x, y):", (10, 80), 
                   cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(info, f"({center_x}, {center_y})", (10, 125), 
                   cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Stack: original crop, processed crop, annotated crop, info
        img1 = cv2.copyMakeBorder(crop, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 
                                 value=(127, 127, 127))
        img2 = cv2.copyMakeBorder(cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR), 
                                 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(127, 127, 127))
        img3 = cv2.copyMakeBorder(crop_annotated, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 
                                 value=(127, 127, 127))
        img4 = cv2.copyMakeBorder(info, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 
                                 value=(127, 127, 127))
        
        return cv2.vconcat([img1, img2, img3, img4])
    
    def _save_detection_results(
        self, 
        locations_df: pd.DataFrame,
        diagnostic_images: List[np.ndarray],
        output_path: str,
        median_x: int,
        median_y: int,
        span_x: int,
        span_y: int
    ):
        """
        Save detection results: CSV and diagnostic images.
        
        Args:
            locations_df: DataFrame with detected locations for each frame
            diagnostic_images: List of diagnostic images for each frame
            output_path: Directory to save results
            median_x: Median X coordinate
            median_y: Median Y coordinate
            span_x: Range of X coordinates
            span_y: Range of Y coordinates
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Save location data
        csv_path = os.path.join(
            output_path, 
            f"LED_Location_{self.config.view_name.replace(' ', '_')}.csv"
        )
        locations_df.to_csv(csv_path, index=False)
        
        # Create composite diagnostic image
        full_composite = cv2.hconcat(diagnostic_images)
        
        # Add header with summary
        header = np.zeros((150, full_composite.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, f"{self.config.view_name} LED Detection", 
                   (20, 35), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(header, f"Median Center: ({median_x}, {median_y})", 
                   (20, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(header, f"Span - X: {span_x}px, Y: {span_y}px", 
                   (20, 125), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        
        full_composite = cv2.vconcat([header, full_composite])
        
        # Save composite
        img_path = os.path.join(
            output_path, 
            f"LED_Detection_{self.config.view_name.replace(' ', '_')}.png"
        )
        cv2.imwrite(img_path, full_composite)
        
        print(f"  Saved detection results to: {output_path}")
    
    def extract_led_signal(
        self, 
        video_path: str,
        led_center: Tuple[int, int],
        output_path: str
    ) -> pd.DataFrame:
        """
        Extract Red LED signal from every frame.
        
        Process:
        1. Read each frame
        2. Extract Red channel around LED center
        3. Average over small region (LED center ± delta)
        4. Create clean binary signal by thresholding
        
        Args:
            video_path: Path to video file
            led_center: (x, y) LED center in full frame coordinates
            output_path: Path to save signal CSV
            
        Returns:
            DataFrame with columns: FrameNumber, RedScore, RedScore_Shifted, Video_LED_Signal
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        center_x, center_y = led_center
        delta = self.config.signal_delta
        
        signals = []
        frame_counter = 0
        
        print(f"\nExtracting LED signal from {self.config.view_name}...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_counter % 100 == 0:
                print(f"  Processing frame {frame_counter}...")
            
            # Extract red channel around LED center
            red_region = frame[
                center_y - delta : center_y + delta + 1,
                center_x - delta : center_x + delta + 1,
                2  # Red channel
            ]
            
            # Average red intensity
            red_score = np.round(np.mean(red_region))
            
            signals.append({
                'FrameNumber': frame_counter,
                'RedScore': red_score
            })
            
            frame_counter += 1
        
        cap.release()
        
        # Create DataFrame
        df = pd.DataFrame(signals)
        
        # Create clean binary signal
        # Threshold at midpoint between 25th and 75th percentiles
        threshold = np.mean([
            np.percentile(df['RedScore'], 25),
            np.percentile(df['RedScore'], 75)
        ])
        
        df['RedScore_Shifted'] = df['RedScore'] - threshold
        df['Video_LED_Signal'] = np.sign(df['RedScore_Shifted'])
        
        # Save signal
        os.makedirs(output_path, exist_ok=True)
        csv_path = os.path.join(
            output_path,
            f"LED_Signal_{self.config.view_name.replace(' ', '_')}.csv"
        )
        df.to_csv(csv_path, index=False)
        
        print(f"  Extracted signal from {len(df)} frames")
        print(f"  Saved to: {csv_path}")
        
        return df

config_map = {
    "Long View": LongViewLEDConfig,
    "Top View": TopViewLEDConfig,
}

def process_view(
    video_path: str,
    view_type: str,
    output_path: str
) -> pd.DataFrame:
    """
    Process one video view: detect LED and extract signal.
    
    This is the main entry point for LED detection and signal extraction.
    
    Args:
        video_path: Path to video file
        view_type: One of "Long View", "Top View", or "Side View"
        output_path: Path to save all outputs
        
    Returns:
        DataFrame with LED signal for alignment
    """
    # Handle Side View specially (requires auto-detection)
    if view_type == "Side View":
        print(f"\n{'='*60}")
        print(f"Processing Side View (auto-detecting Side1 vs Side2)")
        print(f"{'='*60}")
        config = auto_detect_side_view(video_path)
    elif view_type in config_map:
        config = config_map[view_type]()
    else:
        raise ValueError(
            f"Unknown view type: {view_type}. "
            f"Must be one of: Long View, Top View, Side View"
        )
    
    # Create detector and process
    detector = LEDDetector(config)
    
    if view_type != "Side View":
        print(f"\n{'='*60}")
        print(f"Processing {view_type}")
        print(f"{'='*60}")
    
    print(f"Resolution: {config.frame_width}x{config.frame_height}")
    print(f"Force Plate Swap: {'YES (FP1 ↔ FP2)' if config.plate_swap else 'NO (standard)'}")
    
    # Find LED location
    led_center = detector.find_led_location(video_path, output_path)
    
    # Extract signal
    signal_df = detector.extract_led_signal(video_path, led_center, output_path)
    
    return signal_df


# Example usage
if __name__ == "__main__":
    print("LED Detection System Example Usage")
    # Example: Process different view types
    
    # Long View (1920x1080, no swap)
    # video_path = "path/to/long_view.MOV"
    # signal_df = process_view(video_path, "Long View", "output_folder")
    
    # Top View (3840x2160, swap FP1/FP2)
    # video_path = "C:\\\Users\\nishk\\Downloads\\20251216_173131_{WPT_NK}_Top.MP4"
    # output_path = "C:\\Users\\nishk\\Downloads"
    # signal_df = process_view(video_path, "Top View", output_path)
    
    # Side View (3840x2160, auto-detect Side1 vs Side2)
    # video_path = "path/to/side_view.MP4"
    # signal_df = process_view(video_path, "Side View", "output_folder")
    
    # print("\nSignal DataFrame:")
    # print(signal_df.head())

# """
# LED Detection and Alignment System for Multiple Camera Views

# This module provides robust LED detection for Long View (Front), Top View, and Short View (Side)
# cameras. Each view has customized templates and crop regions to handle variations in LED 
# appearance and position.

# Author: Modified from original USC Biomechanics code
# Date: December 2025
# """

# import cv2
# import os
# import numpy as np
# import pandas as pd
# from scipy import stats
# from dataclasses import dataclass
# from typing import Tuple, List, Optional


# @dataclass
# class LEDConfig:
#     """
#     Configuration for LED detection in a specific camera view.
    
#     All measurements are in pixels unless otherwise noted.
#     Crop regions define where to search for the LED in the full frame.
    
#     NEW: plate_swap - If True, swap FP1 and FP2 data after alignment
#     """
#     view_name: str
    
#     # Full frame dimensions (standard for all GoPro/main camera videos)
#     frame_width: int = 1920
#     frame_height: int = 1080
    
#     # Crop region for LED search (x0, x1 = horizontal bounds; y0, y1 = vertical bounds)
#     led_crop_x0: int = 0
#     led_crop_x1: int = 0
#     led_crop_y0: int = 0
#     led_crop_y1: int = 0
    
#     # Template for matching (created in create_led_template method)
#     led_template: Optional[np.ndarray] = None
    
#     # Offset from template top-left corner to LED center (in pixels)
#     template_center_offset_x: int = 0
#     template_center_offset_y: int = 0
    
#     # Region around detected center to average for signal extraction
#     signal_delta: int = 3  # Will average (center ± delta) pixels
    
#     # Number of frames to sample for robust location detection
#     num_frames_to_check: int = 11
    
#     # NEW: Force plate orientation flag
#     plate_swap: bool = False  # If True, swap FP1 and FP2 after alignment
    
#     def __post_init__(self):
#         """Create the LED template after initialization"""
#         if self.led_template is None:
#             self.led_template = self.create_led_template()
    
#     def create_led_template(self) -> np.ndarray:
#         """
#         Create LED template for this view. Override in subclasses.
#         """
#         raise NotImplementedError("Subclass must implement create_led_template")
    
#     def get_crop_region(self, frame: np.ndarray) -> np.ndarray:
#         """Extract the crop region from a frame"""
#         return frame[self.led_crop_y0:self.led_crop_y1, 
#                     self.led_crop_x0:self.led_crop_x1, :]
    
#     def process_crop_for_matching(self, crop: np.ndarray) -> np.ndarray:
#         """
#         Process crop for template matching using Blue-Green subtraction.
#         This method:
#         1. Extracts Blue and Green channels
#         2. Subtracts Green from Blue (highlights Blue LED, darkens surroundings)
#         3. Applies blur to reduce noise
        
#         Returns: Processed grayscale image ready for template matching
#         """
#         blue_channel = crop[:, :, 0]
#         green_channel = crop[:, :, 1]
        
#         blue_minus_green = cv2.subtract(blue_channel, green_channel)
#         processed = cv2.blur(blue_minus_green, (10, 10))
        
#         return processed


# class LongViewLEDConfig(LEDConfig):
#     """
#     Configuration for Long View (Front View) camera.
    
#     Force plate orientation: Left = FP1, Right = FP2 (no swap needed)
#     """
#     def __init__(self):
#         super().__init__(
#             view_name="Long View",
#             led_crop_x0=850,
#             led_crop_x1=1050,
#             led_crop_y0=950,
#             led_crop_y1=1080,
#             template_center_offset_x=45,
#             template_center_offset_y=47,
#             plate_swap=False  # Long view: left=FP1, right=FP2 (standard)
#         )
    
#     def create_led_template(self) -> np.ndarray:
#         """Create template for Long View LED.
#           Template structure (71h x 91w pixels):
#         - Black background (value 0)
#         - Bright rectangle for LED block (value 200)
#         - Dark center where actual LEDs are (value 10)
        
#         The LED block is positioned toward the bottom of the template to handle
#         cases where it's near the edge of the frame.
#         """
#         # Main bright rectangle (outer LED block)
#         # (x0, y0), (x1, y1) format
#         cv2.rectangle(template, (20, 27), (71, 68), 200, -1)
        
#         # Dark center (where actual LEDs are - brighter in green channel)
#         cv2.rectangle(template, (42, 30), (49, 45), 10, -1)
        
#         # Blur to match processed images
#         template = cv2.blur(template, (5, 5))
#         return template


# class TopViewLEDConfig(LEDConfig):
#     """
#     Configuration for Top View camera.
    
#     Force plate orientation: When LED is at top-center:
#     - Physical left (in video) = FP2
#     - Physical right (in video) = FP1
#     So we SWAP to match Long View convention
#     """
#     def __init__(self):
#         super().__init__(
#             view_name="Top View",
#             led_crop_x0=700,
#             led_crop_x1=1000,
#             led_crop_y0=700,
#             led_crop_y1=1080,
#             template_center_offset_x=45,
#             template_center_offset_y=44,
#             plate_swap=True  # Top view: swap FP1/FP2 to match Long View orientation
#         )
    
#     def create_led_template(self) -> np.ndarray:
#         """Create template for Top View LED."""
#         template = np.zeros((77, 91), dtype=np.uint8)
#         cv2.rectangle(template, (20, 20), (71, 57), 200, -1)
#         cv2.rectangle(template, (43, 27), (47, 32), 10, -1)
#         template = cv2.blur(template, (5, 5))
#         return template


# class Side1ViewLEDConfig(LEDConfig):
#     """
#     Configuration for Side1 View (Side View) camera.
    
#     Force plate orientation: 
#     - Near plate (LED on left side of frame) = FP2
#     - Far plate = FP1
#     So we SWAP to match Long View convention
#     """
#     def __init__(self):
#         super().__init__(
#             view_name="Side1 View",
#             # TODO: Adjust these crop values based on actual Side1 footage
#             led_crop_x0=100,   # LED on left side
#             led_crop_x1=400,
#             led_crop_y0=400,
#             led_crop_y1=800,
#             template_center_offset_x=45,
#             template_center_offset_y=47,
#             plate_swap=True  # Side1: near=FP2, far=FP1, so swap
#         )
    
#     def create_led_template(self) -> np.ndarray:
#         """Create template for Side1 View LED."""
#         # Using Long View template as starting point
#         # TODO: Customize based on actual Side1 footage
#         template = np.zeros((71, 91), dtype=np.uint8)
#         cv2.rectangle(template, (20, 27), (71, 68), 200, -1)
#         cv2.rectangle(template, (42, 30), (49, 45), 10, -1)
#         template = cv2.blur(template, (5, 5))
#         return template


# class Side2ViewLEDConfig(LEDConfig):
#     """
#     Configuration for Side2 View (Side View) camera.
    
#     Force plate orientation:
#     - Near plate (LED on right side of frame) = FP1
#     - Far plate = FP2
#     This matches Long View convention (no swap needed)
#     """
#     def __init__(self):
#         super().__init__(
#             view_name="Side2 View",
#             # TODO: Adjust these crop values based on actual Side2 footage
#             led_crop_x0=1520,  # LED on right side
#             led_crop_x1=1820,
#             led_crop_y0=400,
#             led_crop_y1=800,
#             template_center_offset_x=45,
#             template_center_offset_y=47,
#             plate_swap=False  # Side2: near=FP1, far=FP2 (standard)
#         )
    
#     def create_led_template(self) -> np.ndarray:
#         """Create template for Side2 View LED."""
#         # Using Long View template as starting point
#         # TODO: Customize based on actual Side2 footage
#         template = np.zeros((71, 91), dtype=np.uint8)
#         cv2.rectangle(template, (20, 27), (71, 68), 200, -1)
#         cv2.rectangle(template, (42, 30), (49, 45), 10, -1)
#         template = cv2.blur(template, (5, 5))
#         return template


# class LEDDetector:
#     """
#     Detects LED location in video and extracts alignment signal.
#     """
    
#     def __init__(self, config: LEDConfig):
#         """
#         Initialize detector with view-specific configuration.
        
#         Args:
#             config: LEDConfig subclass with view-specific parameters
#         """
#         self.config = config
    
#     def find_led_location(
#         self, 
#         video_path: str, 
#         output_path: str
#     ) -> Tuple[int, int]:
#         """
#         Find LED center location by sampling multiple frames.
#         Find LED center location by sampling multiple frames.
        
#         Strategy:
#         1. Sample frames evenly distributed across video
#         2. Find LED in each frame using template matching
#         3. Use median X and median Y as final location (robust to outliers)
#         4. Save diagnostic images showing detection in each frame
        
#         Args:
#             video_path: Path to video file
#             output_path: Path to save diagnostic images
            
#         Returns:
#             (center_x, center_y): LED center in full frame coordinates
#         """
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError(f"Cannot open video: {video_path}")
        
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         locations_df = pd.DataFrame(
#             columns=['FrameNumber', 'CenterX_Crop', 'CenterY_Crop', 
#                     'CenterX_Full', 'CenterY_Full']
#         )
        
#         diagnostic_images = []
        
#         frame_indices = np.linspace(
#             0, total_frames - 1, 
#             self.config.num_frames_to_check, 
#             dtype=int
#         )
        
#         for frame_idx in frame_indices:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#             ret, frame = cap.read()
            
#             if not ret:
#                 print(f"Warning: Could not read frame {frame_idx}")
#                 continue
            
#             crop = self.config.get_crop_region(frame)
#             processed = self.config.process_crop_for_matching(crop)
            
#             result = cv2.matchTemplate(
#                 processed, 
#                 self.config.led_template, 
#                 cv2.TM_SQDIFF
#             )
#             min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
#             center_x_crop = min_loc[0] + self.config.template_center_offset_x
#             center_y_crop = min_loc[1] + self.config.template_center_offset_y
            
#             center_x_full = center_x_crop + self.config.led_crop_x0
#             center_y_full = center_y_crop + self.config.led_crop_y0
            
#             locations_df = pd.concat([
#                 locations_df,
#                 pd.DataFrame([{
#                     'FrameNumber': frame_idx,
#                     'CenterX_Crop': center_x_crop,
#                     'CenterY_Crop': center_y_crop,
#                     'CenterX_Full': center_x_full,
#                     'CenterY_Full': center_y_full
#                 }])
#             ], ignore_index=True)
            
#             diagnostic = self._create_diagnostic_image(
#                 crop, processed, center_x_crop, center_y_crop, frame_idx
#             )
#             diagnostic_images.append(diagnostic)
        
#         cap.release()
        
#         median_x = int(locations_df['CenterX_Full'].median())
#         median_y = int(locations_df['CenterY_Full'].median())
        
#         span_x = int(locations_df['CenterX_Full'].max() - locations_df['CenterX_Full'].min())
#         span_y = int(locations_df['CenterY_Full'].max() - locations_df['CenterY_Full'].min())
        
#         print(f"\n{self.config.view_name} LED Detection Results:")
#         print(f"  Median location: ({median_x}, {median_y})")
#         print(f"  X span: {span_x} pixels, Y span: {span_y} pixels")
#         print(f"  Force plate swap: {'YES' if self.config.plate_swap else 'NO'}")
        
#         if span_x > 20 or span_y > 20:
#             print(f"  WARNING: Large variation in detected location!")
        
#         self._save_detection_results(
#             locations_df, diagnostic_images, output_path, 
#             median_x, median_y, span_x, span_y
#         )
        
#         return median_x, median_y
    
#     def _create_diagnostic_image(
#         self, 
#         crop: np.ndarray, 
#         processed: np.ndarray,
#         center_x: int, 
#         center_y: int, 
#         frame_num: int
#     ) -> np.ndarray:
#         """Create diagnostic image showing detection for one frame"""
#         crop_annotated = crop.copy()
#         cv2.line(crop_annotated, (center_x, center_y - 20), 
#                 (center_x, center_y + 20), (0, 0, 255), 2)
#         cv2.line(crop_annotated, (center_x - 20, center_y), 
#                 (center_x + 20, center_y), (0, 0, 255), 2)
        
#         info = np.zeros((150, crop.shape[1], 3), dtype=np.uint8)
#         cv2.putText(info, f"Frame: {frame_num}", (10, 35), 
#                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
#         cv2.putText(info, f"Center (x, y):", (10, 80), 
#                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
#         cv2.putText(info, f"({center_x}, {center_y})", (10, 125), 
#                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        
#         img1 = cv2.copyMakeBorder(crop, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 
#                                  value=(127, 127, 127))
#         img2 = cv2.copyMakeBorder(cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR), 
#                                  2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(127, 127, 127))
#         img3 = cv2.copyMakeBorder(crop_annotated, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 
#                                  value=(127, 127, 127))
#         img4 = cv2.copyMakeBorder(info, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 
#                                  value=(127, 127, 127))
        
#         return cv2.vconcat([img1, img2, img3, img4])
    
#     def _save_detection_results(
#         self, 
#         locations_df: pd.DataFrame,
#         diagnostic_images: List[np.ndarray],
#         output_path: str,
#         median_x: int,
#         median_y: int,
#         span_x: int,
#         span_y: int
#     ):
#         """Save detection results: CSV, diagnostic images"""
#         os.makedirs(output_path, exist_ok=True)
        
#         csv_path = os.path.join(
#             output_path, 
#             f"LED_Location_{self.config.view_name.replace(' ', '_')}.csv"
#         )
#         locations_df.to_csv(csv_path, index=False)
        
#         full_composite = cv2.hconcat(diagnostic_images)
        
#         header = np.zeros((150, full_composite.shape[1], 3), dtype=np.uint8)
#         cv2.putText(header, f"{self.config.view_name} LED Detection", 
#                    (20, 35), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
#         cv2.putText(header, f"Median Center: ({median_x}, {median_y})", 
#                    (20, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
#         cv2.putText(header, f"Span - X: {span_x}px, Y: {span_y}px", 
#                    (20, 125), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        
#         full_composite = cv2.vconcat([header, full_composite])
        
#         img_path = os.path.join(
#             output_path, 
#             f"LED_Detection_{self.config.view_name.replace(' ', '_')}.png"
#         )
#         cv2.imwrite(img_path, full_composite)
        
#         print(f"  Saved detection results to: {output_path}")
    
#     def extract_led_signal(
#         self, 
#         video_path: str,
#         led_center: Tuple[int, int],
#         output_path: str
#     ) -> pd.DataFrame:
#         """
#         Extract Red LED signal from every frame.
#         Process:
#         1. Read each frame
#         2. Extract Red channel around LED center
#         3. Average over small region (LED center ± delta)
#         4. Create clean binary signal by thresholding
        
#         Args:
#             video_path: Path to video file
#             led_center: (x, y) LED center in full frame coordinates
#             output_path: Path to save signal CSV
#         Returns:
#             DataFrame with columns: FrameNumber, RedScore, RedScore_Shifted, Video_LED_Signal
#         """
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError(f"Cannot open video: {video_path}")
        
#         center_x, center_y = led_center
#         delta = self.config.signal_delta
        
#         signals = []
#         frame_counter = 0
        
#         print(f"\nExtracting LED signal from {self.config.view_name}...")
        
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             if frame_counter % 100 == 0:
#                 print(f"  Processing frame {frame_counter}...")
            
#             red_region = frame[
#                 center_y - delta : center_y + delta + 1,
#                 center_x - delta : center_x + delta + 1,
#                 2  # Red channel
#             ]
            
#             red_score = np.round(np.mean(red_region))
            
#             signals.append({
#                 'FrameNumber': frame_counter,
#                 'RedScore': red_score
#             })
            
#             frame_counter += 1
        
#         cap.release()
        
#         df = pd.DataFrame(signals)
        
#         threshold = np.mean([
#             np.percentile(df['RedScore'], 25),
#             np.percentile(df['RedScore'], 75)
#         ])
        
#         df['RedScore_Shifted'] = df['RedScore'] - threshold
#         df['Video_LED_Signal'] = np.sign(df['RedScore_Shifted'])
        
#         os.makedirs(output_path, exist_ok=True)
#         csv_path = os.path.join(
#             output_path,
#             f"LED_Signal_{self.config.view_name.replace(' ', '_')}.csv"
#         )
#         df.to_csv(csv_path, index=False)
        
#         print(f"  Extracted signal from {len(df)} frames")
#         print(f"  Saved to: {csv_path}")
        
#         return df


# def process_view(
#     video_path: str,
#     view_type: str,
#     output_path: str
# ) -> pd.DataFrame:
#     """
#     Process one video view: detect LED and extract signal.
    
#     Args:
#         video_path: Path to video file
#         view_type: One of "Long View", "Top View", "Side1 View", or "Side2 View"
#         output_path: Path to save all outputs
        
#     Returns:
#         DataFrame with LED signal for alignment
#     """
#     config_map = {
#         "Long View": LongViewLEDConfig,
#         "Top View": TopViewLEDConfig,
#         "Side1 View": Side1ViewLEDConfig,
#         "Side2 View": Side2ViewLEDConfig,
#     }
    
#     if view_type not in config_map:
#         raise ValueError(
#             f"Unknown view type: {view_type}. "
#             f"Must be one of {list(config_map.keys())}"
#         )
    
#     config = config_map[view_type]()
#     detector = LEDDetector(config)
    
#     print(f"\n{'='*60}")
#     print(f"Processing {view_type}")
#     print(f"{'='*60}")
#     print(f"Force Plate Swap: {'YES (FP1 ↔ FP2)' if config.plate_swap else 'NO (standard)'}")
    
#     led_center = detector.find_led_location(video_path, output_path)
#     signal_df = detector.extract_led_signal(video_path, led_center, output_path)
    
#     return signal_df

