import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample

def cross_correlate(data_long, data_short):
    correlations = []

    for i in range(len(data_long) - len(data_short) + 1):
        window = data_long[i : i + len(data_short)]

        correlations.append(np.dot(data_short, window))

    correlations = np.array(correlations)
    return correlations

def calc_average(array):
    sum = 0.0
    for i in range(len(array)):
        sum += array[i] / len(array)
    return sum

def shift_data(data, shift):
    for i in range(len(data)):
        data[i] -= shift
    return data

def adjust(sample_rate_long, data_long, sample_rate_short, data_short):
    if(sample_rate_long != sample_rate_short):
        if(sample_rate_long > sample_rate_short):
            num_samples = int(len(data_long) * sample_rate_short / sample_rate_long)
            data_long = resample(data_long, num_samples)
        else:
            num_samples = int(len(data_short) * sample_rate_long / sample_rate_short)
            data_short = resample(data_short, num_samples)
    
    data_long = shift_data(data_long, calc_average(data_long))
    data_short = shift_data(data_short, calc_average(data_short))

    return data_long, data_short

def quick_plot(array, x_label="Index", y_label="Value", title="Array Plot"):
    """
    Plots a Python array or NumPy array.

    Parameters:
        array (list or np.array): The array to plot.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(array, marker='o', color='blue', label='Array Values')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def runAudioSync(name_long, name_short):
        sample_rate_long, data_long = wavfile.read(name_long)
        sample_rate_short, data_short = wavfile.read(name_short)

        data_long = data_long.astype(np.float32)
        data_short = data_short.astype(np.float32)

        data_long, data_short = adjust(sample_rate_long, data_long, sample_rate_short, data_short)

        correlations = cross_correlate(data_long, data_short)

        # Print out time of max correlation
        print(np.argmax(correlations) / sample_rate_long, "seconds")

        quick_plot(correlations, x_label = "Shift", y_label = "Amplitude", title = "Cross Correlation Graph")
        quick_plot(data_long, x_label="Samples", y_label = "Amplitude", title = "Long .wav Graph")
        quick_plot(data_short, x_label="Samples", y_label = "Amplitude", title = "Short .wav Graph")

        return np.argmax(correlations) / sample_rate_long;

# runAudioSync("5_sec_wav.wav", "1_sec_wav.wav")