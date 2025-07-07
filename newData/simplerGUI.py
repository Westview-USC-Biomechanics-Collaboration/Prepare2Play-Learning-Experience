"""
Vector Overlay GUI Application

This application provides a GUI for using force plate data to create vector overlays on videos.

Features:
- Select video and force data files (CSV file)
- Choose view mode (Long, Top, Short)
- Live preview or save processed video with vector overlays
- Adjustable lag (based of ledSyncing) for synchronizing video and force data
- Runs overlay processing in a background thread to keep the GUI responsive
- User-friendly file selection and progress feedback

Usage:
- Run this script to launch the GUI
- Select video and force data files
- Select corner of the force plates in the video
- Follow on-screen instructions for corner selection and processing
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import threading

class VectorOverlayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vector Overlay Generator")
        self.root.geometry("600x400")
        self.root.configure(bg="#f0f0f0")

        # File paths
        self.video_path = None
        self.data_path = None
        
        # GUI variables
        self.view_option = tk.StringVar(value="Long")
        self.output_mode = tk.StringVar(value="Live Preview")
        self.output_path = tk.StringVar()
        
        # Status variables
        self.video_status = tk.StringVar(value="No video selected")
        self.data_status = tk.StringVar(value="No data file selected")
        
        self.create_widgets()
        self.center_window()

    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (600 // 2)
        y = (self.root.winfo_screenheight() // 2) - (400 // 2)
        self.root.geometry(f"600x400+{x}+{y}")

    def create_widgets(self):
        # Title
        title_frame = tk.Frame(self.root, bg="#f0f0f0")
        title_frame.pack(pady=10)
        
        tk.Label(title_frame, text="USC Biomechanics Vector Overlay", 
                font=("Arial", 16, "bold"), bg="#f0f0f0").pack()
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(padx=20, pady=10, fill="both", expand=True)
        
        # File selection section
        file_frame = tk.LabelFrame(main_frame, text="📁 File Selection", 
                                  font=("Arial", 10, "bold"), bg="#f0f0f0")
        file_frame.pack(fill="x", pady=(0, 10))
        
        # Video selection
        video_frame = tk.Frame(file_frame, bg="#f0f0f0")
        video_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Button(video_frame, text="🎥 Browse Video", 
                 command=self.browse_video, width=15,
                 bg="#4CAF50", fg="white", font=("Arial", 9, "bold")).pack(side="left")
        
        tk.Label(video_frame, textvariable=self.video_status, 
                bg="#f0f0f0", font=("Arial", 9)).pack(side="left", padx=(10, 0))
        
        # Data selection
        data_frame = tk.Frame(file_frame, bg="#f0f0f0")
        data_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Button(data_frame, text="📊 Browse Data", 
                 command=self.browse_data, width=15,
                 bg="#2196F3", fg="white", font=("Arial", 9, "bold")).pack(side="left")
        
        tk.Label(data_frame, textvariable=self.data_status, 
                bg="#f0f0f0", font=("Arial", 9)).pack(side="left", padx=(10, 0))
        
        # Settings section
        settings_frame = tk.LabelFrame(main_frame, text="⚙️ Settings", 
                                      font=("Arial", 10, "bold"), bg="#f0f0f0")
        settings_frame.pack(fill="x", pady=(0, 10))
        
        # View mode
        view_frame = tk.Frame(settings_frame, bg="#f0f0f0")
        view_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(view_frame, text="View Mode:", font=("Arial", 9, "bold"), 
                bg="#f0f0f0").pack(side="left")
        
        view_combo = ttk.Combobox(view_frame, textvariable=self.view_option, 
                                 values=["Long", "Top", "Short"], state="readonly", width=15)
        view_combo.pack(side="left", padx=(10, 0))
        
        # Output mode
        output_frame = tk.Frame(settings_frame, bg="#f0f0f0")
        output_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(output_frame, text="Output Mode:", font=("Arial", 9, "bold"), 
                bg="#f0f0f0").pack(side="left")
        
        output_combo = ttk.Combobox(output_frame, textvariable=self.output_mode, 
                                   values=["Live Preview", "Save Video"], state="readonly", width=15)
        output_combo.pack(side="left", padx=(10, 0))
        output_combo.bind("<<ComboboxSelected>>", self.on_output_mode_change)
        
        # Output path (initially hidden)
        self.output_path_frame = tk.Frame(settings_frame, bg="#f0f0f0")
        
        tk.Label(self.output_path_frame, text="Output Path:", font=("Arial", 9, "bold"), 
                bg="#f0f0f0").pack(side="left")
        
        tk.Button(self.output_path_frame, text="📁 Choose", 
                 command=self.browse_output, width=10,
                 bg="#FF9800", fg="white", font=("Arial", 8, "bold")).pack(side="left", padx=(10, 0))
        
        tk.Label(self.output_path_frame, textvariable=self.output_path, 
                bg="#f0f0f0", font=("Arial", 8)).pack(side="left", padx=(10, 0))
        
        # Control buttons
        button_frame = tk.Frame(main_frame, bg="#f0f0f0")
        button_frame.pack(fill="x", pady=10)
        
        tk.Button(button_frame, text="🚀 Run Vector Overlay", 
                 command=self.run_overlay, font=("Arial", 12, "bold"),
                 bg="#E91E63", fg="white", height=2).pack(side="left", fill="x", expand=True)
        
        # Info section
        info_frame = tk.LabelFrame(main_frame, text="ℹ️ Info", 
                                  font=("Arial", 9, "bold"), bg="#f0f0f0")
        info_frame.pack(fill="both", expand=True, pady=(10, 0))
        
        info_text = """Supported formats:
• Video: .mp4, .MOV, .avi
• Data: .csv, .xlsx, .xls

View Modes:
• Long: Working
• Top: In progress
• Short: In Progress

Press 'q' during preview to quit early."""
        
        tk.Label(info_frame, text=info_text, justify="left", 
                font=("Arial", 8), bg="#f0f0f0").pack(padx=10, pady=5, anchor="w")

    def on_output_mode_change(self, event=None):
        """Show/hide output path selection based on mode"""
        if self.output_mode.get() == "Save Video":
            self.output_path_frame.pack(fill="x", padx=10, pady=5)
        else:
            self.output_path_frame.pack_forget()

    def browse_video(self):
        """Browse for video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.MOV *.avi *.mkv"),
                ("MP4 files", "*.mp4"),
                ("MOV files", "*.MOV"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.video_path = file_path
            filename = Path(file_path).name
            self.video_status.set(f"{filename}")

    def browse_data(self):
        """Browse for data file (CSV or Excel)"""
        file_path = filedialog.askopenfilename(
            title="Select Force Data File",
            filetypes=[
                ("Data files", "*.csv *.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.data_path = file_path
            filename = Path(file_path).name
            self.data_status.set(f"✅ {filename}")

    def browse_output(self):
        """Browse for output video path"""
        file_path = filedialog.asksaveasfilename(
            title="Save Output Video As",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.output_path.set(file_path)

    def get_auto_output_path(self, video_path, output_folder="processedVideo", extension=".mp4"):
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created or confirmed folder: {output_dir.resolve()}")
        
        base_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_processed_{timestamp}{extension}"
        print(f"Output video path:!")
        return output_dir / output_filename

    def load_data_file(self, file_path):
        """Load data from CSV or Excel file with proper format detection"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.csv':
                # Try to read CSV - check if it has the expected headers
                df = pd.read_csv(file_path)
                
                # Check if this looks like the force plate data format
                expected_cols = ['Fx', 'Fy', 'Fz', '|Ft|', 'Ax', 'Ay']
                if not all(col in df.columns for col in expected_cols):
                    messagebox.showwarning("CSV Format", 
                        f"CSV may not contain all expected force plate columns.\n"
                        f"Expected: {expected_cols}\n"
                        f"Found: {list(df.columns)[:10]}...\n"
                        f"Continuing anyway...")
                
                return df
                
            elif file_ext in ['.xlsx', '.xls']:
                # For Excel files, try the original format first (skip 19 rows)
                try:
                    df = pd.read_excel(file_path, skiprows=19)
                    return df
                except:
                    # If that fails, try reading normally
                    df = pd.read_excel(file_path)
                    return df
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            raise Exception(f"Error loading data file: {str(e)}")

    def run_overlay(self):
        """Run the vector overlay visualization"""
        # Validate inputs
        if not self.video_path or not self.data_path:
            messagebox.showerror("Missing Files", 
                "Please select both video and data files.")
            return

        # If Save Video mode, use auto output path instead of user selection
        if self.output_mode.get() == "Save Video":
            output_path = self.get_auto_output_path(self.video_path)
            # Update the GUI label so user can see where it is saving
            self.output_path.set(str(output_path))
        else:
            output_path = None

        try:
            # Show progress
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Processing...")
            progress_window.geometry("300x100")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            tk.Label(progress_window, text="Loading data and initializing...", 
                    font=("Arial", 10)).pack(pady=20)
            
            progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
            progress_bar.pack(pady=10, padx=20, fill="x")
            progress_bar.start()
            
            self.root.update()

            
            def overlay_task():
                try:
                    self.run_direct_integration(progress_window, output_path)
                except Exception as e:
                    try:
                        progress_window.destroy()
                    except:
                        pass
                    messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

            # Start the overlay in a background thread
            threading.Thread(target=overlay_task, daemon=True).start()

        except Exception as e:
            try:
                progress_window.destroy()
            except:
                pass
            
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")


    def run_direct_integration(self, progress_window, output_path=None):
        """Direct integration approach (fallback)"""
        # Load data
        df = self.load_data_file(self.data_path)
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise Exception("Could not open video file")

        # Import and create VectorOverlay 
        try:
            # Import VectorOverlay class
            import sys
            sys.path.append(r"C:\Users\berke\OneDrive\Desktop\USCBiomechanicsProject\Prepare2Play-Learning-Experience")
            from vector_overlay.vectoroverlay_GUI import VectorOverlay
        except ImportError as e:
            raise Exception(f"VectorOverlay module not found: {str(e)}")

        # Create VectorOverlay instance
        overlay = VectorOverlay(df, cap)
        
        progress_window.destroy()

        # Determine output path
        # output_path is passed in here
        
        # Run the appropriate overlay method
        view_mode = self.view_option.get().lower()
        
        # Show instructions for corner selection
        #messagebox.showinfo("Corner Selection", 
        ##    "Next, you'll select the force plate corners on the video.\n\n"
        ##    "Click on the 8 corners in this order:\n"
         #   "Plate 1: top-left, top-right, bottom-right, bottom-left\n"
        #    "Plate 2: top-left, top-right, bottom-right, bottom-left\n\n"
         #   "Press any key when done selecting all 8 points.")
        
        if view_mode == "long":
            # Use frame-accurate skipping for lag in LongVectorOverlay
            lag = -71  # Or get from user input if needed
            overlay.LongVectorOverlay(outputName=str(output_path) if output_path else None, show_preview=True, lag=lag)
        elif view_mode == "top":
            overlay.TopVectorOverlay(outputName=str(output_path) if output_path else None, lag = -71)
        elif view_mode == "short":
            overlay.ShortVectorOverlay(outputName=str(output_path) if output_path else None, lag = -71)
        else:
            raise ValueError("Invalid view mode selected")

        # Show success message
        if output_path:
            messagebox.showinfo("Success", f"Video saved successfully!\n{output_path}")
        else:
            messagebox.showinfo("Complete", "Live preview finished!")

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = VectorOverlayApp(root)
    

    # Handle window closing
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            cv2.destroyAllWindows()  # Clean up any OpenCV windows
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

# --- Add this to VectorOverlayApp class ---
# In VectorOverlayApp.__init__, add:
# self.lagValue = 0

# In run_direct_integration, replace lag = lagValue with:
# lag = getattr(self, 'lagValue', 0)
# and use lag in overlay.LongVectorOverlay(..., lag=lag)

if __name__ == "__main__":
    main()
