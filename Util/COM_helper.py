class COM_helper:
    def __init__(self, path: str = "pose_landmarks.csv"):
        self.file_path = path
        self.columns = [
            'landmark_0_x', 'landmark_0_y', 'landmark_0_visibility', 
            'landmark_1_x', 'landmark_1_y', 'landmark_1_visibility',
            # ... (full column list)
            'frame_index', 'COM_x', 'COM_y'
        ]
        self.total_rows = None  # Will be set when file is first read
        
    def _count_rows(self):
        """Count total rows in CSV file."""
        if self.total_rows is None:
            try:
                with open(self.file_path, 'r') as f:
                    self.total_rows = sum(1 for line in f) - 1  # Subtract header
                print(f"[COM_helper] CSV has {self.total_rows} data rows")
            except FileNotFoundError:
                print(f"[COM_helper] File not found: {self.file_path}")
                self.total_rows = 0
        return self.total_rows

    def read_line(self, index):
        """
        Read a specific line from the CSV file.
        
        Args:
            index: 0-based row index (not including header)
            
        Returns:
            Parsed line data as list of dicts
            
        Raises:
            IndexError: If index is out of bounds
        """
        total = self._count_rows()
        
        # Check bounds
        if index < 0 or index >= total:
            raise IndexError(
                f"Line index {index} out of range (valid: 0-{total-1})"
            )
        
        # Read the specific line (account for header)
        target_line_num = index + 1  # +1 for header
        
        try:
            with open(self.file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i == target_line_num:
                        return self.__parse_line(line.strip().split(","))
        except FileNotFoundError:
            raise IndexError(f"File not found: {self.file_path}")
        
        raise IndexError(f"Could not read line {index}")

    def __parse_line(self, line):
        """Parse CSV line into structured data."""
        parsed_line = []
        index = 0
        
        # Parse landmark coordinates (33 landmarks x 3 values each)
        while index < len(line) - 3:  # Leave last 3 for frame_index, COM_x, COM_y
            try:
                parsed_line.append({
                    "x": float(line[index]) if line[index] else 0.0,
                    "y": float(line[index + 1]) if line[index + 1] else 0.0,
                    "visibility": float(line[index + 2]) if line[index + 2] else 0.0
                })
                index += 3
            except (ValueError, IndexError):
                # Handle malformed data
                parsed_line.append({"x": 0.0, "y": 0.0, "visibility": 0.0})
                index += 3
        
        # Parse COM data (last 2 values after frame_index)
        try:
            com_x = float(line[-2]) if len(line) >= 2 and line[-2] else 0.0
            com_y = float(line[-1]) if len(line) >= 1 and line[-1] else 0.0
            parsed_line.append({"name": "COM", "x": com_x, "y": com_y})
        except (ValueError, IndexError):
            parsed_line.append({"name": "COM", "x": 0.0, "y": 0.0})
        
        return parsed_line

    def drawFigure(self, frame, row: int):
        """
        Draw COM point on frame.
        
        Args:
            frame: Video frame (numpy array)
            row: 0-based frame index to read from CSV
            
        Returns:
            Frame with COM drawn (or original if error)
        """
        if frame is None:
            print("[COM_helper] Received None frame")
            return frame
        
        frame = frame.copy()  # Prevent in-place modification
        height, width = frame.shape[:2]

        try:
            linedata = self.read_line(row)
        except IndexError as e:
            # Don't print every error, just occasionally
            if row % 100 == 0:
                print(f"[COM_helper] Cannot read row {row}: {e}")
            return frame  # Return unmodified frame
        
        if not linedata:
            return frame

        for point in linedata:
            try:
                x = int(width * float(point['x']))
                y = int(height * float(point['y']))
                
                # Only draw COM point (red)
                if 'name' in point and point['name'] == 'COM':
                    import cv2
                    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                # Optionally draw landmarks (green, smaller)
                # else:
                #     cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            except (KeyError, ValueError, TypeError) as e:
                # Skip invalid points silently
                continue
        
        return frame
    
    def get_frame_count(self):
        """Get total number of frames available in CSV."""
        return self._count_rows()


if __name__ == "__main__":
    import cv2
    com = COM_helper()
    
    # Test with a specific frame
    cam = cv2.VideoCapture(r"path/to/video.mp4")
    ret, frame = cam.read()
    
    if ret:
        try:
            frame_with_com = com.drawFigure(frame, 5)
            cv2.imshow("Frame with COM", frame_with_com)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error: {e}")
            
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
