import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def plot_wav_files(file1, file2, highlight_time=None):
    """
    Plots the waveforms of two WAV files and optionally highlights a specific point on the x-axis.
    
    Parameters:
        file1 (str): Path to the first WAV file.
        file2 (str): Path to the second WAV file.
        highlight_time (float): Time in seconds to highlight on the x-axis.
    """
    # Read the WAV files
    rate1, data1 = wavfile.read(file1)
    rate2, data2 = wavfile.read(file2)

    # Convert to mono if stereo
    if data1.ndim > 1:
        data1 = data1.mean(axis=1)
    if data2.ndim > 1:
        data2 = data2.mean(axis=1)

    # Normalize audio data for consistent comparison
    # data1 = data1 / np.max(np.abs(data1))
    # data2 = data2 / np.max(np.abs(data2))

    # Create time axes
    time1 = np.linspace(0, len(data1) / rate1, num=len(data1))
    time2 = np.linspace(0, len(data2) / rate2, num=len(data2))

    # Find the index for the highlight time, if provided
    if highlight_time is not None:
        highlight_index1 = int(highlight_time * rate1) if highlight_time * rate1 < len(data1) else None
        highlight_index2 = int(highlight_time * rate2) if highlight_time * rate2 < len(data2) else None
    else:
        highlight_index1 = highlight_index2 = None

    # Plot the waveforms
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time1, data1, label=f'File 1: {file1}', color='blue')
    if highlight_index1 is not None:
        plt.scatter(time1[highlight_index1], data1[highlight_index1], color='red', label=f'Highlight: {highlight_time}s')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Waveform of File 1')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time2, data2, label=f'File 2: {file2}', color='green')
    if highlight_index2 is not None:
        plt.scatter(time2[highlight_index2], data2[highlight_index2], color='red', label=f'Highlight: {highlight_time}s')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Waveform of File 2')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage
file1 = '1_sec_wav.wav'
file2 = '5_sec_wav.wav'  
highlight_time = 3.01125  
plot_wav_files(file1, file2, highlight_time)
