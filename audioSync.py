import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate

def calculateConfidence(long_data, short_data, startIndex):
    confidence = 0.0
    alpha = 0.125
    threshold = 0.2

    for i in range(startIndex, startIndex + len(short_data)):
        confidence *= (1 - alpha)
        if abs(short_data[i - startIndex] - long_data[i]) <= threshold:
            confidence += alpha
        # print("Curr Confidence", confidence)
    
    return confidence

# long_array = [4, 1, 2, 3, 4, 5]
# short_array = [1, 2, 3]

# for i in range (0, len(long_array) - len(short_array) + 1):
#     print(i)
#     print(calculateConfidence(long_array, short_array, i))
#     print("___")

def compare_wav_numpy(long_file, short_file):
    # Read the longer and shorter audio files
    sample_rate_long, data_long = wavfile.read(long_file)
    sample_rate_short, data_short = wavfile.read(short_file)

    # Ensure both files have the same sample rate
    if sample_rate_long != sample_rate_short:
        raise ValueError("Sample rates of the files must be the same.")

    # If stereo, take only one channel
    if len(data_long.shape) > 1:
        data_long = data_long[:, 0]
    if len(data_short.shape) > 1:
        data_short = data_short[:, 0]

    # Perform cross-correlation
    correlation = correlate(data_long, data_short, mode='valid')
    max_corr = np.max(correlation)

    # Normalize the correlation to compare
    norm_corr = max_corr / (np.linalg.norm(data_long) * np.linalg.norm(data_short))

    # Define a threshold to decide if the short file is in the long file
    threshold = 0.5  # Adjust based on your requirements
    return norm_corr > threshold, norm_corr

# Example usage
long_file = "5_sec_wav.wav"
short_file = "1_sec_wav.wav"
is_match, correlation_value = compare_wav_numpy(long_file, short_file)
print(f"Match Found: {is_match}, Correlation Value: {correlation_value}")
