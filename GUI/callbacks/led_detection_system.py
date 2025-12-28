"""
LED Detection and Alignment System for Multiple Camera Views

This module provides robust LED detection for Long View (Front), Top View, and Short View (Side)
cameras. Each view has customized templates and crop regions to handle variations in LED 
appearance and position.

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
    """
    view_name: str
    
    # Full frame dimensions (standard for all GoPro/main camera videos)
    frame_width: int = 1920
    frame_height: int = 1080
    
    # Crop region for LED search (x0, x1 = horizontal bounds; y0, y1 = vertical bounds)
    # The LED should always appear within this region
    led_crop_x0: int = 0
    led_crop_x1: int = 0
    led_crop_y0: int = 0
    led_crop_y1: int = 0
    
    # Template for matching (created in create_led_template method)
    led_template: Optional[np.ndarray] = None
    
    # Offset from template top-left corner to LED center (in pixels)
    template_center_offset_x: int = 0
    template_center_offset_y: int = 0
    
    # Region around detected center to average for signal extraction
    signal_delta: int = 3  # Will average (center ± delta) pixels
    
    # Number of frames to sample for robust location detection
    num_frames_to_check: int = 11
    
    def __post_init__(self):
        """Create the LED template after initialization"""
        if self.led_template is None:
            self.led_template = self.create_led_template()
    
    def create_led_template(self) -> np.ndarray:
        """
        Create LED template for this view. Override in subclasses.
        Template represents what we expect the LED block to look like after
        Blue-Green channel subtraction.
        """
        raise NotImplementedError("Subclass must implement create_led_template")
    
    def get_crop_region(self, frame: np.ndarray) -> np.ndarray:
        """Extract the crop region from a frame"""
        return frame[self.led_crop_y0:self.led_crop_y1, 
                    self.led_crop_x0:self.led_crop_x1, :]
    
    def process_crop_for_matching(self, crop: np.ndarray) -> np.ndarray:
        """
        Process crop for template matching using Blue-Green subtraction.
        
        This method:
        1. Extracts Blue and Green channels
        2. Subtracts Green from Blue (highlights Blue LED, darkens surroundings)
        3. Applies blur to reduce noise
        
        Returns: Processed grayscale image ready for template matching
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
    
    LED typically appears in the lower portion of the frame.
    Crop region is generous to handle camera position variations.
    """
    def __init__(self):
        super().__init__(
            view_name="Long View",
            # Crop region - LED expected in lower-center portion of frame
            led_crop_x0=850,   # Start 850 pixels from left edge
            led_crop_x1=1050,  # End 1050 pixels from left edge (200px wide)
            led_crop_y0=950,   # Start 950 pixels from top (near bottom)
            led_crop_y1=1080,  # End at bottom of frame (130px tall)
            # Template offsets (where center is relative to template top-left)
            template_center_offset_x=45,
            template_center_offset_y=47,
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
        """
        template = np.zeros((71, 91), dtype=np.uint8)
        
        # Main bright rectangle (outer LED block)
        # (x0, y0), (x1, y1) format
        cv2.rectangle(template, (20, 27), (71, 68), 200, -1)
        
        # Dark center (where actual LEDs are - brighter in green channel)
        cv2.rectangle(template, (42, 30), (49, 45), 10, -1)
        
        # Blur to match processed images
        template = cv2.blur(template, (5, 5))
        
        return template


class TopViewLEDConfig(LEDConfig):
    """
    Configuration for Top View camera.
    
    LED typically appears in the lower portion of the frame, similar to Long View
    but with slightly different appearance due to viewing angle.
    """
    def __init__(self):
        super().__init__(
            view_name="Top View",
            # Crop region - LED expected in lower portion
            led_crop_x0=700,   # Start 700 pixels from left
            led_crop_x1=1000,  # End 1000 pixels from left (300px wide)
            led_crop_y0=700,   # Start 700 pixels from top
            led_crop_y1=1080,  # End at bottom of frame (380px tall)
            # Template offsets
            template_center_offset_x=45,
            template_center_offset_y=44,  # 39 + 5 to shift away from LED center
        )
    
    def create_led_template(self) -> np.ndarray:
        """
        Create template for Top View LED.
        
        Similar structure to Long View but with slightly different dimensions
        to account for viewing angle differences.
        """
        template = np.zeros((77, 91), dtype=np.uint8)
        
        # Main bright rectangle
        cv2.rectangle(template, (20, 20), (71, 57), 200, -1)
        
        # Dark center where LEDs are
        cv2.rectangle(template, (43, 27), (47, 32), 10, -1)
        
        # Blur to match processed images
        template = cv2.blur(template, (5, 5))
        
        return template


class ShortViewLEDConfig(LEDConfig):
    """
    Configuration for Short View (Side View) camera.
    
    LED appearance and position may differ significantly from other views.
    This configuration should be tuned based on actual Short View footage.
    """
    def __init__(self):
        super().__init__(
            view_name="Short View",
            # Crop region - ADJUST THESE based on where LED appears in Short View
            # Current values are placeholders
            led_crop_x0=800,
            led_crop_x1=1100,
            led_crop_y0=900,
            led_crop_y1=1080,
            # Template offsets - ADJUST based on Short View LED block size
            template_center_offset_x=45,
            template_center_offset_y=47,
        )
    
    def create_led_template(self) -> np.ndarray:
        """
        Create template for Short View LED.
        
        TODO: This template should be customized based on actual Short View footage.
        Current implementation uses Long View template as a starting point.
        
        To customize:
        1. Examine several frames from Short View videos
        2. Look at the LED block after Blue-Green subtraction
        3. Adjust template dimensions and pattern to match
        """
        # Starting with Long View template - CUSTOMIZE THIS
        template = np.zeros((71, 91), dtype=np.uint8)
        
        cv2.rectangle(template, (20, 27), (71, 68), 200, -1)
        cv2.rectangle(template, (42, 30), (49, 45), 10, -1)
        template = cv2.blur(template, (5, 5))
        
        return template


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
        
        if span_x > 20 or span_y > 20:
            print(f"  WARNING: Large variation in detected location!")
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
        """Create diagnostic image showing detection for one frame"""
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
        """Save detection results: CSV, diagnostic images"""
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
        view_type: One of "Long View", "Top View", or "Short View"
        output_path: Path to save all outputs
        
    Returns:
        DataFrame with LED signal for alignment
    """
    # Select configuration based on view
    config_map = {
        "Long View": LongViewLEDConfig,
        "Top View": TopViewLEDConfig,
        "Short View": ShortViewLEDConfig
    }
    
    if view_type not in config_map:
        raise ValueError(
            f"Unknown view type: {view_type}. "
            f"Must be one of {list(config_map.keys())}"
        )
    
    # Create configuration and detector
    config = config_map[view_type]()
    detector = LEDDetector(config)
    
    print(f"\n{'='*60}")
    print(f"Processing {view_type}")
    print(f"{'='*60}")
    
    # Find LED location
    led_center = detector.find_led_location(video_path, output_path)
    
    # Extract signal
    signal_df = detector.extract_led_signal(video_path, led_center, output_path)
    
    return signal_df


# Example usage
if __name__ == "__main__":
    # Example: Process a Long View video
    video_path = "path/to/your/long_view_video.MOV"
    output_path = "path/to/output/folder"
    
    signal_df = process_view(video_path, "Long View", output_path)
    
    print("\nSignal DataFrame:")
    print(signal_df.head())