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
import threading
import os
import pandas as pd
from vector_overlay.vectoroverlay_GUI import VectorOverlay
from GUI.callbacks.ledSyncing import new_led
from Util.video_trimmer import VideoTrimmer
import cv2

def vectorOverlayWithAlignmentCallback(self):
    """
    SEQUENTIAL processing: COM first, then vector overlay with COM drawing.
    This is the correct and fastest approach.
    """
    
    # Prevent multiple concurrent processing
    if hasattr(self, '_processing_thread') and self._processing_thread is not None:
        if self._processing_thread.is_alive():
            print("[WARNING] Processing already in progress!")
            self._pop_up("Processing already in progress!\nPlease wait.")
            return
    
    def threadTarget():
        try:
            print("[INFO] Starting vector overlay with COM processing...")
            parent_path = os.path.dirname(self.Video.path)
            video_file = os.path.basename(self.Video.path)
            force_file = os.path.basename(self.Force.path)

            selected = self.selected_view.get()

            # ========== STEP 1: ALIGNMENT ==========
            print("\n[STEP 1/6] Aligning video and force data...")
            lag, df_aligned = new_led(self, selected, parent_path, video_file, force_file)
            self.state.df_aligned = df_aligned
            
            with open("lag.txt", "w") as f:
                f.write(str(lag))

            print(f"  ✓ Lag detected: {lag} frames")

            # ========== STEP 2: CALCULATE TRIM BOUNDARIES ==========
            print("\n[STEP 2/6] Calculating trim boundaries...")
            
            try:
                trimmer = VideoTrimmer(
                    video_path=self.Video.path,
                    force_data=df_aligned,
                    force_threshold=50.0,
                    step_size=1
                )
                
                start_frame, end_frame = trimmer.calculate_trim_boundaries()
                self.Video.trim_start_frame = start_frame
                self.Video.trim_end_frame = end_frame
                self.Video.has_trim_boundaries = True
                
                duration = (end_frame - start_frame) / self.Video.fps
                print(f"  ✓ Trim boundaries: frames {start_frame}-{end_frame} ({duration:.2f}s)")
                
            except Exception as e:
                print(f"  ⚠ Could not calculate trim boundaries: {e}")
                start_frame = 0
                end_frame = self.Video.total_frames - 1
                self.Video.has_trim_boundaries = False

            # ========== STEP 3: PREPARE FORCE DATA ==========
            print("\n[STEP 3/6] Filtering and renumbering force data...")
            
            force_analysis_filename = force_file.replace('.txt', '_Analysis_Force.csv')
            passed_force_file = os.path.join(parent_path, force_analysis_filename)

            if not os.path.exists(passed_force_file):
                print(f"  ✗ Error: Force analysis file not found at {passed_force_file}")
                return

            force_data = pd.read_csv(passed_force_file)

            # Filter to trim range and renumber
            df_trimmed = df_aligned[
                (df_aligned['FrameNumber'] >= start_frame) & 
                (df_aligned['FrameNumber'] <= end_frame)
            ].copy()
            
            df_trimmed['OriginalFrameNumber'] = df_trimmed['FrameNumber']
            df_trimmed['FrameNumber'] = df_trimmed['FrameNumber'] - start_frame
            
            print(f"  ✓ Prepared {len(df_trimmed)} frames of force data")

            # ========== STEP 4: COM PROCESSING (MUST COMPLETE FIRST!) ==========
            print("\n[STEP 4/6] Computing COM (this may take 30-60 seconds)...")
            
            from vector_overlay.stick_figure_COM import Processor
            from GUI.callbacks.global_variable import globalVariable
            
            sex = globalVariable.sex if globalVariable.sex else 'm'
            
            processor = Processor(
                video_path=self.Video.path,
                data_df=df_aligned,
                lag=0,  # Already applied
                output_mp4="temp_com_only.mp4",  # Won't use this
                trim_boundaries=(start_frame, end_frame)
            )
            
            # This creates pose_landmarks.csv with COM data
            processor.SaveToTxt(
                sex=sex,
                filename='pose_landmarks.csv',
                confidencelevel=0.85,
                displayCOM=False
            )
            
            print(f"  ✓ COM calculations complete")
            self.state.com_enabled = True

            # ========== STEP 5: VECTOR OVERLAY WITH COM ==========
            print("\n[STEP 5/6] Creating vector overlay with COM...")
            
            temp_video = "vector_overlay_temp.mp4"

            v = VectorOverlay(data=force_data, video=self.Video.cam)

            # Process with COM drawing in same pass
            if selected == "Long View":
                v.check_corner("Long View")
                v.LongVectorOverlay_WithCOM(
                    df_trimmed=df_trimmed,
                    outputName=temp_video,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    apply_com=True,
                    com_helper=self.COM_helper
                )
            elif selected == "Short View":
                v.check_corner("Short View")
                v.ShortVectorOverlay_WithCOM(
                    df_trimmed=df_trimmed,
                    outputName=temp_video,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    apply_com=True,
                    com_helper=self.COM_helper
                )
            elif selected == "Top View":
                v.check_corner("Top View")
                v.TopVectorOverlay_WithCOM(
                    df_trimmed=df_trimmed,
                    outputName=temp_video,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    apply_com=True,
                    com_helper=self.COM_helper
                )
            else:
                print(f"  ✗ Unknown view: {selected}")
                return

            print(f"  ✓ Vector overlay with COM complete")

            # ========== STEP 6: UPDATE UI ==========
            print("\n[STEP 6/6] Updating UI...")
            
            trimmed_frame_count = len(df_trimmed)
            self.state.update_slider_bounds(0, trimmed_frame_count - 1)
            self.slider.config(from_=0, to=trimmed_frame_count - 1)
            self.state.is_using_trimmed = True
            
            self.Video.vector_cam = cv2.VideoCapture(temp_video)
            self.Video.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self.Video.vector_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.canvasManager.photo_image3 = self.frameConverter.cvToPillow(camera=self.Video.vector_cam)
            self.canvasManager.canvas3.create_image(200, 150, image=self.canvasManager.photo_image3, anchor="center")

            self.state.vector_overlay_enabled = True
            
            print("\n✓ ALL STEPS COMPLETE!\n")
            self.master.after(0, lambda: self._pop_up("Vector overlay with COM complete!"))
            
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.master.after(0, lambda: self._pop_up(f"Error during processing:\n{str(e)}"))
        
        finally:
            self._processing_thread = None

    # Launch processing thread
    self._processing_thread = threading.Thread(target=threadTarget, daemon=True)
    self._processing_thread.start()
    
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
