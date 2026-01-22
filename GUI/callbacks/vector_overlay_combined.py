import threading
import os
import pandas as pd
import cv2
from vector_overlay.vectoroverlay_GUI import VectorOverlay
from GUI.callbacks.ledSyncing_with_detection_system import new_led  
from GUI.callbacks import global_variable
from Util.force_boundary_finder import find_force_boundaries, get_trimmed_subset
from vector_overlay.com_processor_modified import BoundaryProcessor as Processor
import warnings

# Suppress specific deprecation warnings
warnings.filterwarnings(
    "ignore", 
    message="SymbolDatabase.GetPrototype() is deprecated.*"
)

# Configurable settings
MAX_COM_WORKERS = 6  # Easily adjustable number of workers for COM calculation
FORCE_THRESHOLD = 50  # Minimum force in Newtons to include in processing
BOUNDARY_PADDING = 10  # Extra frames before/after force threshold
SHOW_LANDMARKS = False  # Show green landmark dots (set to True to enable)
USE_DETECTION_SYSTEM = True  # Use new LED detection system (set to False for original method)


def vectorOverlayWithAlignmentCallback(self):
    def threadTarget():
        print("[INFO] Starting Vector Overlay with Alignment...")
        parent_path = os.path.dirname(self.Video.path)
        video_file = os.path.basename(self.Video.path)
        force_file = os.path.basename(self.Force.path)

        print(f"Video file: {video_file}")
        print(f"Force file: {force_file}")

        selected = self.selected_view.get()

        cap = cv2.VideoCapture(self.Video.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        ret, frame  = cap.read()
        new_filename = "First_Frame_of_Video.PNG"
        cv2.imwrite(os.path.join(parent_path, new_filename), frame)
        cap.release()


        # ======================================================================
        # STEP 1: ALIGN VIDEO AND FORCE DATA
        # ======================================================================
        print("\n[STEP 1] Aligning video and force data...")
        lag, df_aligned = new_led(
            self, selected, parent_path, video_file, force_file, 
            use_detection_system=USE_DETECTION_SYSTEM
        )
        print("Raw df_aligned columns from new_led:", df_aligned.columns.tolist())

        # Rename columns to match what VectorOverlay expects
        column_rename = {
            'Fx1': 'FP1_Fx', 'Fy1': 'FP1_Fy', 'Fz1': 'FP1_Fz', 'Ft1': 'FP1_|F|', 
            'Ax1': 'FP1_Ax', 'Ay1': 'FP1_Ay',
            'Fx2': 'FP2_Fx', 'Fy2': 'FP2_Fy', 'Fz2': 'FP2_Fz', 'Ft2': 'FP2_|F|', 
            'Ax2': 'FP2_Ax', 'Ay2': 'FP2_Ay',
            'Fx3': 'FP3_Fx', 'Fy3': 'FP3_Fy', 'Fz3': 'FP3_Fz', 'Ft3': 'FP3_|F|', 
            'Ax3': 'FP3_Ax', 'Ay3': 'FP3_Ay',
            'abs time (s)': 'Time(s)'  # Also ensure Time(s) column exists
        }
        df_aligned.rename(columns=column_rename, inplace=True)
        
        self.state.df_aligned = df_aligned
        
        # with open("lag.txt", "w") as f:
        #     f.write(str(lag))

        print(f"Alignment complete. Lag: {lag} frames")
        print(f"df_aligned shape: {df_aligned.shape}")
        print(f"df_aligned columns: {list(df_aligned.columns)}")
        print("[DEBUG] Current Sex in globalVariable:", global_variable.globalVariable.sex)
        # ======================================================================
        # STEP 2: FIND FORCE BOUNDARIES (TRIMMING)
        # ======================================================================
        print("\n[STEP 2] Finding force boundaries for trimming...")
        try:
            boundary_start, boundary_end = find_force_boundaries(
                df_aligned, 
                threshold=FORCE_THRESHOLD,
                padding_frames=BOUNDARY_PADDING
            )
            
            # Store boundaries in state for future reference
            self.state.boundary_start = boundary_start
            self.state.boundary_end = boundary_end
            
            print(f"Processing subset: frames {boundary_start} to {boundary_end}")
            
        except Exception as e:
            print(f"[ERROR] Failed to find force boundaries: {e}")
            print("[INFO] Using full frame range as fallback")
            boundary_start = int(df_aligned['FrameNumber'].min())
            boundary_end = int(df_aligned['FrameNumber'].max())
        # ======================================================================
        # STEP 3: RUN COM CALCULATION ON TRIMMED SUBSET
        # ======================================================================
        print("\n[STEP 3] Running COM calculation on trimmed subset...")
        com_csv_path = os.path.join(parent_path, "pose_landmarks.csv")
        
        try:
            # Determine sex for COM calculation
            print("[DEBUG] Current Sex in globalVariable:", global_variable.globalVariable.sex)
            sex = global_variable.globalVariable.sex if global_variable.globalVariable.sex else 'm'  # default to male

            if not global_variable.globalVariable.sex:
                print("[WARNING] Sex not set in globalVariable, defaulting to 'male'")
            
            # Create COM processor using BoundaryProcessor wrapper
            processor = Processor(self.Video.path)
            
            print("Processor class:", Processor)
            print("SaveToTxt signature:", Processor.SaveToTxt.__code__.co_varnames) 

            # Run COM calculation with boundaries
            processor.SaveToTxt(
                sex=sex,
                filename=com_csv_path,
                confidencelevel=0.85,
                displayCOM=True,
                start_frame=boundary_start,
                end_frame=boundary_end,
                max_workers=MAX_COM_WORKERS
            )
            
            if com_csv_path and os.path.exists(com_csv_path):
            # Update COM_helper to use the new CSV
                from Util.COM_helper import COM_helper
                self.COM_helper = COM_helper(com_csv_path)
                print(f"[INFO] COM_helper updated with: {com_csv_path}")
            
        except Exception as e:
            print(f"[ERROR] COM calculation failed: {e}")
            import traceback
            traceback.print_exc()
            com_csv_path = None

        # ======================================================================
        # STEP 4: RUN VECTOR OVERLAY WITH COM ON TRIMMED SUBSET
        # ======================================================================
        print("\n[STEP 4] Running vector overlay with COM visualization...")
        
        # Get trimmed force data
        df_trimmed = get_trimmed_subset(df_aligned, boundary_start, boundary_end)
        
        # Prepare output filename
        temp_video = "vector_overlay_temp.mp4"
        
        try:
            # Reset video to start
            self.Video.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Create VectorOverlay instance
            v = VectorOverlay(data=df_trimmed, video=self.Video.cam, view=selected)
            
            # Detect corners for the selected view
            v.check_corner(selected)
                        
            # Rename columns in df_trimmed to match VectorOverlay expectations
            column_rename = {
                'Fx1': 'FP1_Fx', 'Fy1': 'FP1_Fy', 'Fz1': 'FP1_Fz', 'Ft1': 'FP1_|F|', 
                'Ax1': 'FP1_Ax', 'Ay1': 'FP1_Ay',
                'Fx2': 'FP2_Fx', 'Fy2': 'FP2_Fy', 'Fz2': 'FP2_Fz', 'Ft2': 'FP2_|F|', 
                'Ax2': 'FP2_Ax', 'Ay2': 'FP2_Ay',
                'Fx3': 'FP3_Fx', 'Fy3': 'FP3_Fy', 'Fz3': 'FP3_Fz', 'Ft3': 'FP3_|F|', 
                'Ax3': 'FP3_Ax', 'Ay3': 'FP3_Ay',
                'abs time (s)': 'Time(s)'  # ensure time column is consistent
            }
            df_trimmed.rename(columns=column_rename, inplace=True)
            self.state.df_trimmed = df_trimmed.reset_index(drop=True)
            
            print("[COLUMNS BEFORE] LongVectorOverlay:", list(df_trimmed.columns))
            print("Path to COM CSV before VectorOverlay:", com_csv_path)
            # Run vector overlay with COM for the selected view
            if selected == "Long View":
                v.LongVectorOverlay(
                    df_aligned=df_trimmed,
                    outputName=temp_video,
                    lag=lag,
                    com_csv_path=com_csv_path,
                    show_landmarks=SHOW_LANDMARKS,
                    boundary_start=boundary_start,
                    boundary_end=boundary_end
                )
            elif selected == "Side1 View":
                v.SideVectorOverlay(
                    df_aligned=df_trimmed,
                    outputName=temp_video,
                    lag=lag,
                    com_csv_path=com_csv_path,
                    show_landmarks=SHOW_LANDMARKS,
                    boundary_start=boundary_start,
                    boundary_end=boundary_end,
                    is_side1=True  # FP1 is near
                )
            elif selected == "Side2 View":
                v.SideVectorOverlay(
                    df_aligned=df_trimmed,
                    outputName=temp_video,
                    lag=lag,
                    com_csv_path=com_csv_path,
                    show_landmarks=SHOW_LANDMARKS,
                    boundary_start=boundary_start,
                    boundary_end=boundary_end,
                    is_side1=False  # FP2 is near
                )
            elif selected == "Top View":
                v.TopVectorOverlay(
                    df_aligned=df_trimmed,
                    outputName=temp_video,
                    lag=lag,
                    com_csv_path=com_csv_path,
                    show_landmarks=SHOW_LANDMARKS,
                    boundary_start=boundary_start,
                    boundary_end=boundary_end
                )
            
            print(f"[INFO] Vector overlay complete: {temp_video}")
            
            # Load the result video
            self.Video.vector_cam = cv2.VideoCapture(temp_video)
            self.Video.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Display on canvas 3
            self.Video.vector_cam.set(cv2.CAP_PROP_POS_FRAMES, self.state.loc)
            self.canvasManager.photo_image3 = self.frameConverter.cvToPillow(
                camera=self.Video.vector_cam
            )
            self.canvasManager.canvas3.create_image(
                200, 150, 
                image=self.canvasManager.photo_image3, 
                anchor="center"
            )
            
            self.state.vector_overlay_enabled = True
            print("[SUCCESS] All processing complete!")
            
        except Exception as e:
            print(f"[ERROR] Vector overlay failed: {e}")
            import traceback
            traceback.print_exc()

    # Launch in thread
    threading.Thread(target=threadTarget, daemon=True).start()
##--------------------------OLD CODE UNDER---------------------------##
# import threading
# import os
# import pandas as pd
# from vector_overlay.vectoroverlay_GUI import VectorOverlay
# from vector_overlay.COM_vectoroverlay import Processor
# from GUI.callbacks.ledSyncing import run_led_syncing  # rename or move if needed
# from GUI.callbacks.ledSyncing import new_led
# from GUI.callbacks.global_variable import globalVariable
# import cv2

# def vectorOverlayWithAlignmentCallback(self):
#     def threadTarget():
#         print("[INFO] Running LED syncing and vector overlay...")
#         parent_path = os.path.dirname(self.Video.path)
#         video_file = os.path.basename(self.Video.path)
#         force_file = os.path.basename(self.Force.path)

#         print(f"Name of the video file: {video_file}")
#         print(f"Name of the force file: {force_file}")

#         selected = self.selected_view.get()

#         # Step 1: Run syncing and get lag
#         #lag = run_led_syncing(self, parent_path, video_file, force_file)
#         lag, df_aligned = new_led(self, selected, parent_path, video_file, force_file)
#         self.state.df_aligned = df_aligned
        
#         with open("lag.txt", "w") as f:
#             f.write(str(lag))

#         print(f"Detected lag: {lag} frames")

#         force_analysis_filename = force_file.replace('.txt', '_Analysis_Force.csv')
#         passed_force_file = os.path.join(parent_path, force_analysis_filename)

#         print(f"Looking for force file at: {passed_force_file}")

#         # Check if file exists
#         if not os.path.exists(passed_force_file):
#             print(f"Error: Force analysis file not found at {passed_force_file}")
#             return

#         force_data = pd.read_csv(passed_force_file)

#         # Step 3: Run vector overlay with adjusted force data
#         self.Video.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

#         temp_video = "vector_overlay_temp.mp4"

#         v = VectorOverlay(data=force_data, video=self.Video.cam)
#         # v = Processor(self.Video.path, self.Force.data, lag, temp_video)

#         if selected == "Long View":
#             v.check_corner("Long View")
#             v.LongVectorOverlay(df_aligned, outputName=temp_video, lag=lag)
#             # v.SaveToTxt(sex=globalVariable.sex, filename="pose_landmarks.csv", confidencelevel=0.85, displayCOM=True)
#         elif selected == "Short View":
#             v.check_corner("Short View")
#             v.ShortVectorOverlay(df_aligned, outputName=temp_video, lag=lag)
#             # v.SaveToTxt(sex=globalVariable.sex, filename="pose_landmarks.csv", confidencelevel=0.85, displayCOM=True)
#         elif selected == "Top View":
#             v.check_corner("Top View")
#             v.TopVectorOverlay(df_aligned, outputName=temp_video, lag=lag)
#             # v.SaveToTxt(sex=globalVariable.sex, filename="pose_landmarks.csv", confidencelevel=0.85, displayCOM=True)

#         self.Video.vector_cam = cv2.VideoCapture(temp_video)
#         self.Video.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

#         # Step 4: Display overlay result on 3rd canvas
#         self.Video.vector_cam.set(cv2.CAP_PROP_POS_FRAMES, self.state.loc)
#         self.canvasManager.photo_image3 = self.frameConverter.cvToPillow(camera=self.Video.vector_cam)
#         self.canvasManager.canvas3.create_image(200, 150, image=self.canvasManager.photo_image3, anchor="center")

#         self.state.vector_overlay_enabled = True

#     # Launch in thread
#     threading.Thread(target=threadTarget, daemon=True).start()
