import csv

class COM_helper:
    def __init__(self, path: str = "D:\\Backup_download\\pose_landmarks.csv"):
        self.file_path = path
        self._cache = {}
        self._coords_are_normalized = None  # Will auto-detect
        self._load_cache()
    
    def has_data_for_frame(self, video_frame: int) -> bool:
        """
        Returns True if COM data exists for this video frame.
        """
        csv_frame = video_frame + self.video_align

        if "FrameNumber" not in self.df.columns:
            return False

        rows = self.df[self.df["FrameNumber"] == csv_frame]
        return not rows.empty

    def _load_cache(self):
        """Load all COM data into memory for fast access"""
        try:
            with open(self.file_path, 'r') as f:
                reader = csv.DictReader(f)
                
                com_x_values = []
                com_y_values = []
                
                for row in reader:
                    try:
                        # Try different possible column names for frame index
                        frame_idx = None
                        for col_name in ['frame_index', 'FrameNumber', 'Frame']:
                            if col_name in row and row[col_name]:
                                frame_idx = int(float(row[col_name]))
                                break
                        
                        if frame_idx is None:
                            continue
                        
                        com_x = float(row.get('COM_x', 0))
                        com_y = float(row.get('COM_y', 0))
                        
                        # Collect values for auto-detection
                        if com_x != 0 and com_y != 0:
                            com_x_values.append(com_x)
                            com_y_values.append(com_y)
                        
                        self._cache[frame_idx] = {
                            'x': com_x,
                            'y': com_y,
                            'row': row
                        }
                    except (ValueError, KeyError) as e:
                        print(f"[WARN] Failed to parse row: {e}")
                        continue
                
                # Auto-detect if coordinates are normalized (0-1) or pixels (>1)
                if com_x_values and com_y_values:
                    max_x = max(com_x_values)
                    max_y = max(com_y_values)
                    
                    # If values are all between 0 and 1, they're normalized
                    # If they're > 1, they're likely pixel coordinates
                    self._coords_are_normalized = (max_x <= 1.0 and max_y <= 1.0)
                    
                    print(f"[COM_helper] Loaded {len(self._cache)} frames from {self.file_path}")
                    print(f"[COM_helper] Frame range: {min(self._cache.keys())} to {max(self._cache.keys())}")
                    print(f"[COM_helper] COM_x range: {min(com_x_values):.2f} to {max_x:.2f}")
                    print(f"[COM_helper] COM_y range: {min(com_y_values):.2f} to {max_y:.2f}")
                    print(f"[COM_helper] Coordinates are {'NORMALIZED (0-1)' if self._coords_are_normalized else 'PIXEL values'}")
                else:
                    print(f"[COM_helper] Loaded {len(self._cache)} frames but no valid COM data found")
                    self._coords_are_normalized = False
                    
        except Exception as e:
            print(f"[ERROR] Failed to load COM cache: {e}")
            import traceback
            traceback.print_exc()
    
    def read_line(self, frame_index: int):
        """Get COM data for a specific frame index"""
        if frame_index in self._cache:
            return self._cache[frame_index]
        else:
            return {'x': 0, 'y': 0}
    
    def drawFigure(self, frame, row: int):
        """
        Draw COM marker on frame
        
        Args:
            frame: The video frame (numpy array)
            row: The frame index to look up
            
        Returns:
            Modified frame with COM drawn
        """
        import cv2
        
        if frame is None:
            return frame
        
        frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Get COM data for this frame
        com_data = self.read_line(row)
          
        try:
            com_x = float(com_data['x'])
            com_y = float(com_data['y'])
            
            # Skip if no data (0, 0)
            if com_x == 0 and com_y == 0:
                return frame
            
            # Convert to pixel coordinates if necessary
            if self._coords_are_normalized:
                pixel_x = int(com_x * width)
                pixel_y = int(com_y * height)
            else:
                # Already in pixel coordinates
                pixel_x = int(com_x)
                pixel_y = int(com_y)
            
            # Clamp to frame boundaries
            pixel_x = max(0, min(pixel_x, width - 1))
            pixel_y = max(0, min(pixel_y, height - 1))
            
            # Draw COM marker (red circle)
            if pixel_x > 0 and pixel_y > 0:
                cv2.circle(frame, (pixel_x, pixel_y), 12, (0, 0, 255), -1)
                # Only print occasionally to avoid spam
                if row % 30 == 0:
                    print(f"[COM_helper] Frame {row}: Drew COM at pixel ({pixel_x}, {pixel_y}) from raw ({com_x:.2f}, {com_y:.2f})")
            
        except (KeyError, ValueError, TypeError) as e:
            if row % 100 == 0:  # Only print occasionally
                print(f"[ERROR] Failed to draw COM for frame {row}: {e}")
        
        return frame


if __name__ == "__main__":
    import cv2
    import pandas as pd
    
    # First, inspect the CSV
    print("=== Inspecting CSV ===")
    df = pd.read_csv('D:\\Backup_download\\pose_landmarks.csv')
    print("\nColumns:", df.columns.tolist())
    print("\nFirst few rows of COM data:")
    print(df[['frame_index', 'COM_x', 'COM_y']].head(10))
    print("\nCOM value ranges:")
    print(f"COM_x: {df['COM_x'].min():.2f} to {df['COM_x'].max():.2f}")
    print(f"COM_y: {df['COM_y'].min():.2f} to {df['COM_y'].max():.2f}")
    
    # Now test the COM_helper
    print("\n=== Testing COM_helper ===")
    com = COM_helper('D:\\Backup_download\\pose_landmarks.csv')
    
    if com._cache:
        first_frame = min(com._cache.keys())
        print(f"\nTesting with first available frame: {first_frame}")
        
        data = com.read_line(first_frame)
        print(f"Frame {first_frame} COM raw data: x={data['x']}, y={data['y']}")
        
        # Test with video
        cam = cv2.VideoCapture(r"D:\\Backup_download\\Long 4.MOV")
        cam.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        ret, frame = cam.read()
        
        if ret:
            print(f"Video frame size: {frame.shape[1]}x{frame.shape[0]} (width x height)")
            frameWithFigure = com.drawFigure(frame, first_frame)
            cv2.imshow("Frame with COM", frameWithFigure)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        cam.release()

''' Step 3: Run the diagnostic

Run the updated `COM_helper.py` file as `__main__` and share the output. It will tell us:
1. What the actual COM values are in your CSV
2. Whether they're normalized or pixel coordinates
3. Where it's trying to draw the circle

The output should look like:
```
=== Inspecting CSV ===
Columns: [...]
First few rows of COM data:
   frame_index    COM_x    COM_y
0         1424  950.234  540.123
...
COM value ranges:
COM_x: 800.00 to 1100.00
COM_y: 400.00 to 700.00

=== Testing COM_helper ===
[COM_helper] Loaded 1019 frames
[COM_helper] Coordinates are PIXEL values
Frame 1424 COM raw data: x=950.234, y=540.123
Video frame size: 1920x1080
[COM_helper] Frame 1424: Drew COM at pixel (950, 540) from raw (950.23, 540.12)
'''
# if __name__ == "__main__":
#     import cv2
#     import pandas as pd
    
#     # Load and inspect the CSV
#     df = pd.read_csv('D:\\Backup_download\\pose_landmarks.csv')
#     print("\n=== CSV Columns ===")
#     print(df.columns.tolist())
    
#     print("\n=== First few rows of COM data ===")
#     print(df[['frame_index', 'COM_x', 'COM_y']].head())
    
#     print("\n=== COM value ranges ===")
#     print(f"COM_x range: {df['COM_x'].min()} to {df['COM_x'].max()}")
#     print(f"COM_y range: {df['COM_y'].min()} to {df['COM_y'].max()}")
    # import cv2
    
    # com = COM_helper('D:\\Backup_download\\pose_landmarks.csv')
    # if com._cache:
    #     first_frame = min(com._cache.keys())
    #     print(f"\nTesting with first available frame: {first_frame}")
        
    #     # Test reading a specific frame
    #     data = com.read_line(first_frame)
    #     print(f"Frame {first_frame} COM: x={data['x']}, y={data['y']}")
        
    #     # Test drawing on a frame
    #     cam = cv2.VideoCapture(r"D:\\Backup_download\\Long 4.MOV")
        
    #     # Seek to the correct frame
    #     cam.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    #     ret, frame = cam.read()
        
    #     if ret:
    #         originalFrame = frame.copy()
    #         frameWithFigure = com.drawFigure(originalFrame, first_frame)
    #         cv2.imshow("Frame with COM", frameWithFigure)
    #         cv2.imshow("Original Frame", frame)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #     else:
    #         print(f"Could not read frame {first_frame} from video")
        
    #     cam.release()
    # else:
    #     print("No COM data loaded!")
    
###---------------------------OLD CODE BELOW---------------------------###
# class COM_helper:
#     def __init__(self, path:str="pose_landmarks.csv"):
#         self.file_path = path
#         self.columns = ['landmark_0_x', 'landmark_0_y', 'landmark_0_visibility', 'landmark_1_x', 'landmark_1_y', 'landmark_1_visibility', 'landmark_2_x', 'landmark_2_y', 'landmark_2_visibility', 'landmark_3_x', 'landmark_3_y', 'landmark_3_visibility', 'landmark_4_x', 'landmark_4_y', 'landmark_4_visibility', 'landmark_5_x', 'landmark_5_y', 'landmark_5_visibility', 'landmark_6_x', 'landmark_6_y', 'landmark_6_visibility', 'landmark_7_x', 'landmark_7_y', 'landmark_7_visibility', 'landmark_8_x', 'landmark_8_y', 'landmark_8_visibility', 'landmark_9_x', 'landmark_9_y', 'landmark_9_visibility', 'landmark_10_x', 'landmark_10_y', 'landmark_10_visibility', 'landmark_11_x', 'landmark_11_y', 'landmark_11_visibility', 'landmark_12_x', 'landmark_12_y', 'landmark_12_visibility', 'landmark_13_x', 'landmark_13_y', 'landmark_13_visibility', 'landmark_14_x', 'landmark_14_y', 'landmark_14_visibility', 'landmark_15_x', 'landmark_15_y', 'landmark_15_visibility', 'landmark_16_x', 'landmark_16_y', 'landmark_16_visibility', 'landmark_17_x', 'landmark_17_y', 'landmark_17_visibility', 'landmark_18_x', 'landmark_18_y', 'landmark_18_visibility', 'landmark_19_x', 'landmark_19_y', 'landmark_19_visibility', 'landmark_20_x', 'landmark_20_y', 'landmark_20_visibility', 'landmark_21_x', 'landmark_21_y', 'landmark_21_visibility', 'landmark_22_x', 'landmark_22_y', 'landmark_22_visibility', 'landmark_23_x', 'landmark_23_y', 'landmark_23_visibility', 'landmark_24_x', 'landmark_24_y', 'landmark_24_visibility', 'landmark_25_x', 'landmark_25_y', 'landmark_25_visibility', 'landmark_26_x', 'landmark_26_y', 'landmark_26_visibility', 'landmark_27_x', 'landmark_27_y', 'landmark_27_visibility', 'landmark_28_x', 'landmark_28_y', 'landmark_28_visibility', 'landmark_29_x', 'landmark_29_y', 'landmark_29_visibility', 'landmark_30_x', 'landmark_30_y', 'landmark_30_visibility', 'landmark_31_x', 'landmark_31_y', 'landmark_31_visibility', 'landmark_32_x', 'landmark_32_y', 'landmark_32_visibility', 'frame_index', 'COM_x', 'COM_y\n']
  

#     def read_line(self, index):
#         with open(self.file_path, 'r') as f:
#             for i, line in enumerate(f):
#                 if i == index:
#                     return self.__parse_line(line.split(","))
#         raise IndexError("Line index out of range")

    
#     def __parse_line(self, line):
#         index = 0
#         parsed_line = []
#         while index < len(line)-3:
#             item = line[index]
#             parsed_line.append({"x":line[index],"y":line[index+1],"visibility":line[index+2]})
#             index += 3
        
#         parsed_line.append({"name":"COM","x":line[-2],"y":line[-1]})
#         return parsed_line
    
#     def drawFigure(self, frame, row: int): 
#         import cv2
#         frame = frame.copy()  # Prevent in-place modification
        
#         height, width = frame.shape[:2]

#         linedata = self.read_line(row)
#         if not linedata:
#             print("[WARN] No data to draw.")
#             return frame

#         for point in linedata:
#             # print(f"point: {point}")
#             try:
#                 x = int(width * float(point['x']))
#                 y = int(height* float(point['y']))
#                 if 'name' in point:
#                     cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
#                 # else:
#                 #   cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

#             except (KeyError, ValueError, TypeError) as e:
#                 print(f"[ERROR] Skipping invalid point: {point} — {e}")
        
#         # print("[DEBUG] finished drawing")
#         return frame

# if __name__ == "__main__":
#     import cv2
#     com = COM_helper()
#     line = com.read_line('pose_landmarks.csv', 1)
#     # print(line)
#     cam = cv2.VideoCapture(r"C:\Users\chase\Downloads\0519test.mp4")
#     ret, frame = cam.read()
#     if ret:
#         originalFrame = frame.copy()  # Make a copy before any drawing
#         frameWithFigure = com.drawFigure(originalFrame, 5)
#         cv2.imshow("Frame with COM", frameWithFigure)
#         cv2.imshow("Original Frame", frame)  # Show the truly original one
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
