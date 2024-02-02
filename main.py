import numpy as np
import pandas as pd
from scipy.fft import fft, fftshift
from scipy.signal import butter, lfilter, hilbert, sosfilt
import matplotlib.pyplot as plt

# Constants
numADCSamples = 200
numADCBits = 16
numTX = 1
numRX = 4
chirpLoop = 2
Fs = 4e6
c = 3e8
ts = numADCSamples / Fs
slope = 64.985e12
B_valid = ts * slope
detaR = c / (2 * B_valid)
startFreq = 60.25e9
lambda_radar = c / startFreq
isReal = 0

# Read Bin file
filename = 'data/adc_data22.bin'
with open(filename, 'rb') as f:
    adcDataRow = np.fromfile(f, dtype=np.int16)

# Process data depending on ADC bits
if numADCBits != 16:
    l_max = 2 ** (numADCBits - 1) - 1
    adcDataRow[adcDataRow > l_max] -= 2 ** numADCBits

process_num = 1024
fileSize = process_num * 2 * numADCSamples * numTX * numRX * 2
PRTnum = fileSize // (numADCSamples * numRX)
fileSize = PRTnum * numADCSamples * numRX
adcData = adcDataRow[:fileSize]

if isReal:
    numChirps = fileSize // (numADCSamples * numRX)
    # Reshape the real data into a 2D array with each chirp as a column
    LVDS = np.reshape(adcData, (numADCSamples * numRX, numChirps))
    # Transpose to make each row correspond to one chirp
    LVDS = LVDS.T
else:
    # For complex data
    numChirps = fileSize // (2 * numADCSamples * numRX)
    # Pre-allocate array for complex data
    LVDS = np.zeros((1, fileSize // 2), dtype=complex)
    # Combine real and imaginary parts into complex numbers
    counter = 0;
    for i in range(0, fileSize - 2, 4):
        LVDS[0, counter] = adcData[i] + 1j * adcData[i + 2]
        LVDS[0, counter + 1] = adcData[i + 1] + 1j * adcData[i + 3]
        counter += 2
    # Reshape to a 2D array with each chirp as a column
    LVDS = np.reshape(LVDS, (numChirps, numADCSamples * numRX))
    # Transpose to make each row correspond to one chirp

# Reorganize data for each receiving antenna
adcData_reorganized = np.zeros((numRX, numChirps * numADCSamples), dtype=LVDS.dtype)
for row in range(numRX):
    for i in range(numChirps):
        start_idx = i * numADCSamples
        end_idx = start_idx + numADCSamples
        adcData_reorganized[row, start_idx:end_idx] = LVDS[i, row * numADCSamples:(row + 1) * numADCSamples]

retVal = np.reshape(adcData_reorganized[0, :], (numChirps, numADCSamples)).T
process_adc = np.zeros((numADCSamples, numChirps // 2), dtype=np.complex_)
for nchirp in range(0, numChirps, 2):
    process_adc[:, nchirp // 2] = retVal[:, nchirp]

# Perform FFT
fft1d = np.abs(fft(process_adc, axis=0)).T
X, Y = np.meshgrid(np.arange(numADCSamples) * detaR, np.arange(numChirps // 2))

# Plotting Range FFT, 20*log10() 意思是将幅度转换为分贝
plt.figure()
plt.plot(np.arange(numADCSamples) * detaR, 20 * np.log10(fft1d[0, :]))
plt.xlabel('Range (m)')
plt.ylabel('Magnitude (dB)')
plt.title('Range Dimension FFT of a single chirp')

plt.figure()
plt.pcolormesh(X, Y, 20 * np.log10(fft1d))
plt.xlabel('Range (m)')
plt.ylabel('Chirp number')
plt.title('Range Dimension - 1D FFT Result')
plt.colorbar()

"""================================"""

# Set parameters for phase unwrapping
RangFFT = 256
fft_data_last = np.zeros(RangFFT)  # Energy amplitude accumulation
range_max = 0
adcdata = process_adc
numChirps = adcdata.shape[1]  # Number of chirps, updated to 1024

# Perform range-dimension FFT
fft_data = fft(adcdata, n=RangFFT, axis=0)
fft_data = fft_data.T  # Transpose to get 1024 * 256 matrix

# Calculate the magnitude (absolute value) of the complex FFT result
fft_data_abs = np.abs(fft_data)

# Adjust the distance resolution because the sample points have been expanded to 256
deltaR = Fs * c / (2 * slope * RangFFT)
fft_data_abs[:, :10] = 0  # Zeroing to remove DC component

# Generate a 3D plot
fft11d = fft_data_abs

# Create meshgrid for 3D plot
M, N = np.meshgrid(np.arange(RangFFT) * deltaR, np.arange(1, numChirps + 1))

# Plot the 3D mesh
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(M, N, fft11d, cmap='viridis')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Chirp pulse number')
ax.set_zlabel('Magnitude')
ax.set_title('After Static Clutter Removal: Range-Dimension 1D FFT Result')
plt.show()

"""================================"""

# Extract phase (phase arctangent) - 提取相位 (相位反正切)
real_data = np.real(fft_data)  # Real part - 实部
imag_data = np.imag(fft_data)  # Imaginary part - 虚部

# Calculate the phase for each range bin using atan2 for the entire matrix
angle_fft = np.arctan2(imag_data, real_data)

# Range-bin tracking - 找出能量最大的点，即人体的位置
# To find the range bin with the maximum energy
range_max = 0
max_num = 0

# Initialize a storage array for energy accumulation
fft_data_last = np.zeros(RangFFT)

for j in range(RangFFT):
    # Check if the range bin is within the specified distance limits
    if (j * detaR) < 2.5 and (j * detaR) > 0.5:
        # Sum the energies for a non-coherent accumulation
        fft_data_last[j] = np.sum(fft_data_abs[:, j])

        # Update the maximum energy and corresponding bin number
        if fft_data_last[j] > range_max:
            range_max = fft_data_last[j]
            max_num = j

# Extract phase from the range bin with the maximum energy
angle_fft_last = angle_fft[:, max_num]

phi = angle_fft_last  # (n_chirps,)
angle_fft_last = np.unwrap(phi)

# Phase difference 相位差分
angle_fft_diff = np.diff(angle_fft_last)
angle_fft_diff = np.insert(angle_fft_diff, 0, 0)  # Inserting a zero at the beginning to maintain array size


# Moving average filtering
def smoothdata(signal, window_size):
    """
    模仿matlab中的smoothdata函数。
    :param signal:
    :param window_size:
    :return:
    """
    smoothed_signal = np.copy(signal)
    length = len(signal)
    # 对于数组的每一个元素，计算移动平均
    for i in range(length):
        # 确定窗口的起始和结束位置
        start = max(i - window_size // 2, 0)
        end = min(i + window_size // 2 + 1, length)
        # 计算窗口中数据的平均值
        smoothed_signal[i] = np.mean(signal[start:end])

    return smoothed_signal


phi_smooth = smoothdata(angle_fft_diff, 5)

# FFT of the phase signal
N1 = len(phi_smooth)
FS = 20
fft_phase = np.abs(fft(phi_smooth))
f = np.arange(N1) * (FS / N1)

# # Bandpass filter for breath signal
# breath_pass = butter(4, [0.1, 0.5], btype='bandpass', fs=FS)
# breath_data = lfilter(breath_pass[0], breath_pass[1], phi_smooth)

# The sosMatrix and ScaleValues from MATLAB's breath_pass would be provided as follows:
sosMatrix = np.array(
    [[1, 0, -1, 1, -1.96315431309333, 0.964395116007807],
     [1, 0, -1, 1, -1.85009994293252, 0.868089891124299]])

ScaleValues = np.array([0.0601804080654874, 0.0601804080654874, 1])

# Apply the scale values to the sosMatrix
# The last scale value is applied after filtering, so we exclude it here
# 创建一个新的 SOS 矩阵来应用缩放值
sos = np.copy(sosMatrix)
# 只对分子系数 (b) 应用缩放值，分母系数 (a) 保持不变
sos[:, :3] *= ScaleValues[0]  # 假设所有分子系数都应用同一个缩放值
sos[:, 3] = 1  # 确保 a_0 是 1

# 应用 SOS 滤波器到信号
breath_data = sosfilt(sos, phi_smooth)

# 应用最后一个缩放值到输出
breath_data *= ScaleValues[-1]
"""================================"""
# Spectral Estimation - FFT - Peak Interval
N1 = len(breath_data)
fs = 20  # Sampling rate of the breath/heartbeat signal
fshift = np.arange(-N1 / 2, N1 / 2) * (fs / N1)  # Zero-centered frequency range for plotting

# Perform FFT and shift it to center zero frequency
breath_fre = np.abs(fftshift(fft(breath_data)))  # Double-sided spectrum (magnitude)

# Plot the spectrum
plt.figure()
plt.plot(fshift[:N1 // 2], breath_fre[:N1 // 2])  # Plot only the positive frequencies
plt.xlabel('Frequency (f/Hz)')
plt.ylabel('Magnitude')
plt.title('Breath Signal FFT')

# Search for the peak in the spectrum to determine the breath rate
breath_fre_max = np.max(breath_fre)
breath_index = np.argmax(breath_fre)

# Calculate breath rate
breath_count = (fs * (N1 / 2 - (breath_index)) / N1) * 60

# Print the breath rate
print(f"Breath Rate: {breath_count} breaths per minute")

# Show the plot
plt.show()

"""================================"""

# Bandpass filter for heart signal
# heart_pass = butter(8, [0.8, 2.0], btype='bandpass', fs=FS)
# heart_data = lfilter(heart_pass[0], heart_pass[1], phi_smooth)

sosMatrix = np.array(
    [[1, 0, -1, 1, -1.49069622543010, 0.821584856975103],
     [1, 0, -1, 1, -1.85299358095713, 0.916570170374599],
     [1, 0, -1, 1, -1.65884708697245, 0.751234896982056],
     [1, 0, -1, 1, -1.46163518810405, 0.655484575195733]])

ScaleValues = np.array([
    0.175431355008461,
    0.175431355008461,
    0.161866596066365,
    0.161866596066365,
    1
])

# Apply the scale values to the sosMatrix
# The last scale value is applied after filtering, so we exclude it here
# 创建一个新的 SOS 矩阵来应用缩放值
sos = np.copy(sosMatrix)
# 只对分子系数 (b) 应用缩放值，分母系数 (a) 保持不变
sos[:, :3] *= ScaleValues[0]  # 假设所有分子系数都应用同一个缩放值
sos[:, 3] = 1  # 确保 a_0 是 1

# 应用 SOS 滤波器到信号
heart_data = sosfilt(sos, phi_smooth)

# 应用最后一个缩放值到输出
heart_data *= ScaleValues[-1]

"""================================"""
# Heart rate signal processing
N1 = len(heart_data)
fs = 20  # Sampling rate of the heart rate signal
fshift = np.arange(-N1 / 2, N1 / 2) * (fs / N1)  # Zero-centered frequency range
f = np.arange(0, N1) * (fs / N1)  # Frequency range for each point

# Perform FFT and shift
heart_fre = np.abs(fftshift(fft(heart_data)))  # Double-sided spectrum (magnitude)
heart = np.abs(fft(heart_data))  # Single-sided spectrum (magnitude)

# Plot the heart rate signal FFT
plt.figure()
plt.plot(f[:200], heart[:200])  # Only plot the first 200 points for a closer look
plt.xlabel('Frequency (f/Hz)')
plt.ylabel('Magnitude')
plt.title('Heart Rate Signal FFT')

# Search for the maximum in the first half of the spectrum
heart_fre_max = np.max(heart_fre[:N1 // 2])
heart_index = np.argmax(heart_fre[:N1 // 2])

# Determine if a heartbeat is present based on magnitude confidence
if heart_fre_max < 1e-2:
    heart_index = N1  # Set to an invalid index if no heartbeat is detected

# Calculate the heart rate
heart_count = (fs * (N1 / 2 - (heart_index)) / N1) * 60  # Convert to beats per minute

# Print the heart rate
print(f"Heart Rate: {heart_count} beats per minute")

# Show the plot
plt.show()

"""================================"""

# Extract envelope for heart signal
analytic_signal = hilbert(heart_data)
env_envelope = np.abs(analytic_signal)

# Plotting the results
plt.figure()
plt.plot(env_envelope)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Heartbeat Envelope')

# Normalizing heartbeat signal
normalized_heartbeat = heart_data / env_envelope

plt.figure()
plt.plot(normalized_heartbeat)
plt.xlabel('Time (s)')
plt.ylabel('Normalized Amplitude')
plt.title('Normalized Heartbeat Signal')
