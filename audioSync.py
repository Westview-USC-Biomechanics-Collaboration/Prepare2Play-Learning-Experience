import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

# Define the sequences
sample_rate_long, long_data = wavfile.read("5_sec_wav.wav") # np.array([1, 2, 3, 4, 5])
sample_rate_short, short_data = wavfile.read("1_sec_wav.wav") # np.array([1, 2, 3])

# Account for different sampling rates
if sample_rate_short != sample_rate_long:
    if sample_rate_long > sample_rate_short:
        # Resample long to match short rate
        num_samples = int(len(long_data) *  sample_rate_short / sample_rate_long)
        long_data = resample(long_data, num_samples)
    else:
        # Resample short to match long rate
        num_samples = int(len(short_data) * sample_rate_long / sample_rate_short)
        short_data = resample(short_data, num_samples)

# Length of sequences
n_long = len(long_data)
n_short = len(short_data)

# Output variables
max_correlation = 0.0
index_max = 0

# Compute normalized cross-correlation
normalized_correlation = []
for i in range(n_long - n_short + 1):  # Sliding window over long_data
    # Extract the current window from long_data
    window = long_data[i:i + n_short]
    
    # Compute numerator (dot product)
    numerator = np.dot(short_data, window)
    
    # Compute denominator (product of norms)
    denominator = np.sqrt(np.sum(short_data ** 2) * np.sum(window ** 2))
    
    # Compute normalized correlation
    normalized_correlation.append(numerator / denominator)

    if normalized_correlation[-1] > max_correlation:
        max_correlation = normalized_correlation[-1]
        index_max = i

# Convert result to a numpy array
normalized_correlation = np.array(normalized_correlation)

# Print the result
print("Normalized Cross-Correlation:", normalized_correlation)
print("Max Correlation Value was ", max_correlation, "Index at ", index_max, "Occurred at", index_max * 1 / sample_rate_short)
print("Sample Rate Long", sample_rate_long, "Sample Rate Short", sample_rate_short)

# def calculateConfidence(long_data, short_data, startIndex):
#     confidence = 0.0
#     alpha = 0.125
#     threshold = 0.2

#     for i in range(startIndex, startIndex + len(short_data)):
#         confidence *= (1 - alpha)
#         if abs(short_data[i - startIndex] - long_data[i]) <= threshold:
#             confidence += alpha
#         # print("Curr Confidence", confidence)
    
#     return confidence

# long_file = "5_sec_wav.wav"
# short_file = "1_sec_wav.wav"

# sample_rate_long, data_long = wavfile.read(long_file)
# sample_rate_short, data_short = wavfile.read(short_file)

# print(data_long)
# print("length", len(data_long))
# print("___")
# print(data_short)
# print("length", len(data_short))

# long_array = [4, 1, 2, 3, 4, 5]
# short_array = [1, 2, 3]

# for i in range (0, len(long_array) - len(short_array) + 1):
#     print(i)
#     print(calculateConfidence(long_array, short_array, i))
#     print("___")

# def compare_wav_numpy(long_file, short_file):
#     # Read the longer and shorter audio files
#     sample_rate_long, data_long = wavfile.read(long_file)
#     sample_rate_short, data_short = wavfile.read(short_file)

#     # Ensure both files have the same sample rate
#     if sample_rate_long != sample_rate_short:
#         raise ValueError("Sample rates of the files must be the same.")

#     maxConfidence = 0.0
#     index = 0

#     for i in range (0, len(data_long) - len(data_short) + 1):
#         currConfidence = calculateConfidence(data_long, data_short, i)
#         if(currConfidence > maxConfidence):
#             maxConfidence = currConfidence
#             index = i

#     threshold = 0.2
#     return maxConfidence > threshold, maxConfidence, index

# # Example usage
# long_file = "5_sec_wav.wav"
# short_file = "1_sec_wav.wav"
# is_match, correlation_value, index = compare_wav_numpy(long_file, short_file)
# print(f"Match Found: {is_match}, Correlation Value: {correlation_value}, Index: {index}")
