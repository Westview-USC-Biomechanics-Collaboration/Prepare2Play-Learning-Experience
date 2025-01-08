import tkinter as tk
from tkinter import filedialog, messagebox
from enum import Enum, auto
from audioSync import runAudioSync

# Define states for the state machine
class State(Enum):
    RECEIVE_DATA = auto()
    PROCESS_DATA = auto()
    DISPLAY_DATA = auto()

# Initialize state
current_state = State.RECEIVE_DATA
correlation_time = None  # Variable to store the time of max correlation

# State machine functions
def receive_data():
    global current_state
    if not long_file_entry.get() or not short_file_entry.get():
        messagebox.showerror("Error", "Please select both long and short .wav files.")
    else:
        current_state = State.PROCESS_DATA
        process_data()

def process_data():
    global current_state, correlation_time
    long_file = long_file_entry.get()
    short_file = short_file_entry.get()

    try:
        # Modified to capture the time of max correlation
        correlation_time = runAudioSync(long_file, short_file)  # Assuming this method now returns the max correlation time
        current_state = State.DISPLAY_DATA
        display_data()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        current_state = State.RECEIVE_DATA

def display_data():
    global current_state
    message = "AudioSync completed successfully!"
    if correlation_time is not None:
        message += f"\nTime of max correlation: {correlation_time:.2f} seconds"
    messagebox.showinfo("Success", message)
    current_state = State.RECEIVE_DATA

# File selection function
def select_file(entry_widget):
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, file_path)

# Run function to trigger state transitions
def run_state_machine():
    if current_state == State.RECEIVE_DATA:
        receive_data()

# Create main window
root = tk.Tk()
root.title("AudioSync GUI")

# Long file selection
long_file_label = tk.Label(root, text="Select Long .wav File:")
long_file_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")

long_file_entry = tk.Entry(root, width=50)
long_file_entry.grid(row=0, column=1, padx=5, pady=5)

long_file_button = tk.Button(root, text="Browse", command=lambda: select_file(long_file_entry))
long_file_button.grid(row=0, column=2, padx=5, pady=5)

# Short file selection
short_file_label = tk.Label(root, text="Select Short .wav File:")
short_file_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")

short_file_entry = tk.Entry(root, width=50)
short_file_entry.grid(row=1, column=1, padx=5, pady=5)

short_file_button = tk.Button(root, text="Browse", command=lambda: select_file(short_file_entry))
short_file_button.grid(row=1, column=2, padx=5, pady=5)

# Run button
run_button = tk.Button(root, text="Run AudioSync", command=run_state_machine)
run_button.grid(row=2, column=0, columnspan=3, pady=10)

# Start the GUI loop
root.mainloop()
