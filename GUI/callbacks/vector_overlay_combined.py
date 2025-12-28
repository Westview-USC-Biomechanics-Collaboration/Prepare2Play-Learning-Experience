"""
vector_overlay_combined.py - Main callback for vector overlay with COM

This file orchestrates:
1. LED detection and alignment
2. Vector overlay generation 
3. COM calculation (using stick_figure_COM.py)
"""

import threading
import os
import pandas as pd
from vector_overlay.vectoroverlay_GUI import VectorOverlay
from vector_overlay.stick_figure_COM import Processor  # ← Use stick_figure_COM!
from GUI.callbacks.ledSyncing import new_led
from GUI.callbacks.global_variable import globalVariable
import cv2


def vectorOverlayWithAlignmentCallback(self):
    """
    Main entry point for creating vector overlay with COM.
    
    This function:
    1. Runs LED detection to align video and force data
    2. Creates vector overlay using VectorOverlay class
    3. Calculates COM using stick_figure_COM.Processor
    4. Displays result in canvas3
    """
    
    def threadTarget():
        error_message = None
        
        try:
            print("\n" + "="*60)
            print("VECTOR OVERLAY WITH COM - STARTING")
            print("="*60)
            
            # ================================================================
            # STEP 1: Get file paths
            # ================================================================
            print("\n[STEP 1/6] Gathering file information...")
            parent_path = os.path.dirname(self.Video.path)
            video_file = os.path.basename(self.Video.path)
            force_file = os.path.basename(self.Force.path)
            selected = self.selected_view.get()

            print(f"  Video: {video_file}")
            print(f"  Force: {force_file}")
            print(f"  View: {selected}")

            # ================================================================
            # STEP 2: LED Detection and Alignment
            # ================================================================
            print("\n[STEP 2/6] Running LED detection and alignment...")
            print("  This detects the LED in the video and aligns with force data")
            
            lag, df_aligned = new_led(self, selected, parent_path, video_file, force_file)
            self.state.df_aligned = df_aligned
            
            # Save lag for other parts of code
            with open("lag.txt", "w") as f:
                f.write(str(lag))

            print(f"  ✓ Alignment complete! Lag = {lag} frames")
            print(f"  ✓ Aligned data shape: {df_aligned.shape}")

            # ================================================================
            # STEP 3: Verify force file exists
            # ================================================================
            print("\n[STEP 3/6] Verifying force analysis file...")
            force_analysis_filename = force_file.replace('.txt', '_Analysis_Force.csv')
            passed_force_file = os.path.join(parent_path, force_analysis_filename)

            if not os.path.exists(passed_force_file):
                error_message = f"Force analysis file not found: {passed_force_file}"
                raise FileNotFoundError(error_message)

            force_data = pd.read_csv(passed_force_file)
            print(f"  ✓ Force data loaded: {len(force_data)} rows")

            # ================================================================
            # STEP 4: Create Vector Overlay (WITHOUT COM yet)
            # ================================================================
            print("\n[STEP 4/6] Creating vector overlay...")
            print("  This draws force arrows on the video")
            
            # Reset video to beginning
            self.Video.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Temporary video file for vector overlay
            temp_video = "vector_overlay_temp.mp4"

            # Create VectorOverlay instance
            v = VectorOverlay(data=force_data, video=self.Video.cam)
            
            # Detect force plate corners
            print(f"  Detecting corners for {selected}...")
            v.check_corner(selected)
            
            # Create overlay based on view
            if selected == "Long View":
                print("  Creating Long View vector overlay...")
                v.LongVectorOverlay(df_aligned, outputName=temp_video, lag=lag)
            elif selected == "Short View":
                print("  Creating Short View vector overlay...")
                v.ShortVectorOverlay(df_aligned, outputName=temp_video, lag=lag)
            elif selected == "Top View":
                print("  Creating Top View vector overlay...")
                v.TopVectorOverlay(df_aligned, outputName=temp_video, lag=lag)
            
            print(f"  ✓ Vector overlay saved to: {temp_video}")

            # ================================================================
            # STEP 5: Calculate COM using stick_figure_COM
            # ================================================================
            print("\n[STEP 5/6] Computing COM (Center of Mass)...")
            print("  This uses MediaPipe to detect body landmarks")
            print("  and calculates COM for each frame")
            print("  (This may take 30-60 seconds...)")
            
            # Create Processor from stick_figure_COM
            processor = Processor(self.Video.path)
            
            # Run COM calculation
            # This will:
            # - Detect pose landmarks in each frame
            # - Calculate COM based on body segment masses
            # - Save results to pose_landmarks.csv
            processor.SaveToTxt(
                sex=globalVariable.sex if globalVariable.sex else 'm',
                filename="pose_landmarks.csv",
                confidencelevel=0.85,
                displayCOM=True
            )
            
            print("  ✓ COM calculation complete!")
            print("  ✓ Results saved to: pose_landmarks.csv")

            # ================================================================
            # STEP 6: Display result in canvas3
            # ================================================================
            print("\n[STEP 6/6] Loading result into GUI...")
            
            # Load the vector overlay video
            self.Video.vector_cam = cv2.VideoCapture(temp_video)
            self.Video.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Display first frame in canvas3
            self.Video.vector_cam.set(cv2.CAP_PROP_POS_FRAMES, self.state.loc)
            self.canvasManager.photo_image3 = self.frameConverter.cvToPillow(
                camera=self.Video.vector_cam
            )
            self.canvasManager.canvas3.create_image(
                200, 150, 
                image=self.canvasManager.photo_image3, 
                anchor="center"
            )

            # Enable vector overlay flag
            self.state.vector_overlay_enabled = True
            
            print("  ✓ Vector overlay loaded in canvas3")
            print("\n" + "="*60)
            print("✓ COMPLETE! Vector overlay with COM ready.")
            print("="*60 + "\n")
            
            # Success popup
            self.master.after(0, lambda: self._pop_up(
                "✓ Vector overlay with COM completed successfully!\n\n"
                "Results saved to:\n"
                "- vector_overlay_temp.mp4 (video with arrows)\n"
                "- pose_landmarks.csv (COM data)"
            ))

        except Exception as e:
            # Capture exception
            error_message = str(e)
            print(f"\n{'='*60}")
            print(f"✗ ERROR: {error_message}")
            print(f"{'='*60}\n")
            
            import traceback
            traceback.print_exc()
            
            # Show error popup
            self.master.after(0, lambda msg=error_message: self._pop_up(
                f"Error during vector overlay:\n\n{msg}"
            ))

    # Launch processing in background thread
    threading.Thread(target=threadTarget, daemon=True).start()



################################ OLD CODE BELOW ################################
# import threading
# import os
# import pandas as pd
# from vector_overlay.vectoroverlay_GUI import VectorOverlay
# from vector_overlay.COM_vectoroverlay import Processor
# from GUI.callbacks.ledSyncing import run_led_syncing
# from GUI.callbacks.ledSyncing import new_led
# from GUI.callbacks.global_variable import globalVariable
# from Util.video_trimmer import VideoTrimmer
# import cv2
    ################################## Newer Code Below ##################################
# def vectorOverlayWithAlignmentCallback(self):
#     def threadTarget():
#         print("[INFO] Running LED syncing and vector overlay...")
#         parent_path = os.path.dirname(self.Video.path)
#         video_file = os.path.basename(self.Video.path)
#         force_file = os.path.basename(self.Force.path)

#         print(f"Video file: {video_file}")
#         print(f"Force file: {force_file}")

#         selected = self.selected_view.get()

#         # ========== STEP 1: ALIGNMENT ==========
#         print("\n[STEP 1] Aligning video and force data...")
#         lag, df_aligned = new_led(self, selected, parent_path, video_file, force_file)
#         self.state.df_aligned = df_aligned
        
#         with open("lag.txt", "w") as f:
#             f.write(str(lag))

#         print(f"Detected lag: {lag} frames")

#         # ========== STEP 2: CALCULATE TRIM BOUNDARIES ==========
#         print("\n[STEP 2] Calculating trim boundaries from aligned force data...")
        
#         try:
#             # Use df_aligned which has FrameNumber column matching video frames
#             trimmer = VideoTrimmer(
#                 video_path=self.Video.path,
#                 force_data=df_aligned,  # Use ALIGNED data
#                 force_threshold=50.0,
#                 step_size=1  # df_aligned is already at video frame rate (one row per frame)
#             )
            
#             start_frame, end_frame = trimmer.calculate_trim_boundaries()
            
#             # Store boundaries
#             self.Video.trim_start_frame = start_frame
#             self.Video.trim_end_frame = end_frame
#             self.Video.has_trim_boundaries = True
            
#             duration = (end_frame - start_frame) / self.Video.fps
#             print(f"[INFO] Trim boundaries: frames {start_frame}-{end_frame} ({duration:.2f}s)")
            
#         except Exception as e:
#             print(f"[WARNING] Could not calculate trim boundaries: {e}")
#             print("[WARNING] Will process full video")
#             start_frame = 0
#             end_frame = self.Video.total_frames - 1
#             self.Video.has_trim_boundaries = False

#         # ========== STEP 3: CREATE TRIMMED VIDEO ==========
#         print("\n[STEP 3] Creating trimmed video for processing...")
        
#         try:
#             trimmed_video_path = trimmer.create_trimmed_video(
#                 output_path=os.path.join(parent_path, "video_trimmed_aligned.mp4")
#             )
            
#             self.Video.trimmed_path = trimmed_video_path
#             print(f"[INFO] Trimmed video created: {trimmed_video_path}")
            
#             # Use trimmed video for processing
#             processing_video_path = trimmed_video_path
            
#         except Exception as e:
#             print(f"[WARNING] Could not create trimmed video: {e}")
#             print("[WARNING] Using original video")
#             processing_video_path = self.Video.path

#         # ========== STEP 4: PREPARE FORCE DATA ==========
#         force_analysis_filename = force_file.replace('.txt', '_Analysis_Force.csv')
#         passed_force_file = os.path.join(parent_path, force_analysis_filename)

#         if not os.path.exists(passed_force_file):
#             print(f"Error: Force analysis file not found at {passed_force_file}")
#             return

#         force_data = pd.read_csv(passed_force_file)

#         # ========== STEP 5: FILTER df_aligned to TRIMMED RANGE ==========
#         # Create a version of df_aligned that only includes the trimmed frames
#         # AND renumber frames starting from 0 for the trimmed video
#         print(f"\n[STEP 5] Filtering aligned data to trimmed range...")
        
#         df_trimmed = df_aligned[
#             (df_aligned['FrameNumber'] >= start_frame) & 
#             (df_aligned['FrameNumber'] <= end_frame)
#         ].copy()
        
#         # IMPORTANT: Renumber frames to match trimmed video (0-based)
#         df_trimmed['OriginalFrameNumber'] = df_trimmed['FrameNumber']
#         df_trimmed['FrameNumber'] = df_trimmed['FrameNumber'] - start_frame
        
#         print(f"[INFO] Trimmed data: {len(df_trimmed)} rows")
#         print(f"[INFO] Trimmed frame range: 0-{len(df_trimmed)-1} (was {start_frame}-{end_frame})")

#         # ========== STEP 6: RUN VECTOR OVERLAY ==========
#         print("\n[STEP 6] Running vector overlay on trimmed video...")
        
#         # Open the processing video (either trimmed or original)
#         processing_cam = cv2.VideoCapture(processing_video_path)
#         processing_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

#         temp_video = "vector_overlay_temp.mp4"

#         v = VectorOverlay(data=force_data, video=processing_cam)

#         if selected == "Long View":
#             v.check_corner("Long View")
#             v.LongVectorOverlay(df_trimmed, outputName=temp_video, lag=0)  # lag already applied
#         elif selected == "Short View":
#             v.check_corner("Short View")
#             v.ShortVectorOverlay(df_trimmed, outputName=temp_video, lag=0)
#         elif selected == "Top View":
#             v.check_corner("Top View")
#             v.TopVectorOverlay(df_trimmed, outputName=temp_video, lag=0)
#         else:
#             print(f"[ERROR] Unknown view: {selected}")
#             return

#         processing_cam.release()

#         # ========== STEP 7: DISPLAY RESULT ==========
#         self.Video.vector_cam = cv2.VideoCapture(temp_video)
#         self.Video.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

#         # Display overlay result on 3rd canvas
#         self.Video.vector_cam.set(cv2.CAP_PROP_POS_FRAMES, self.state.loc)
#         self.canvasManager.photo_image3 = self.frameConverter.cvToPillow(camera=self.Video.vector_cam)
#         self.canvasManager.canvas3.create_image(200, 150, image=self.canvasManager.photo_image3, anchor="center")

#         self.state.vector_overlay_enabled = True
        
#         print("\n[INFO] Vector overlay complete!")

#     # Launch in thread
#     threading.Thread(target=threadTarget, daemon=True).start()

################################ OLD CODE BELOW ################################

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
