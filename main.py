import numpy as np
from scipy.fft import fft, fftshift
from scipy.signal import hilbert, sosfilt
import matplotlib.pyplot as plt


class Radar_params:
    def __init__(self, numADCSamples, numADCBits, numTX, numRX, chirpLoop, Fs, slope, startFreq, isReal=0):
        self.numADCSamples = numADCSamples  # ADC采样点数
        self.numADCBits = numADCBits  # ADC采样位数
        self.numTX = numTX  # 发射天线的个数
        self.numRX = numRX  # 接收天线的个数
        self.chirpLoop = chirpLoop  # 一帧中chirp的个数
        self.Fs = Fs  # 采样频率
        self.ts = numADCSamples / Fs  # ADC采样时间
        self.c = 3e8
        self.slope = slope  # 斜率
        self.startFreq = startFreq
        self.B_valid = self.ts * slope  # 有效带宽
        self.detaR = self.c / (2 * self.B_valid)  # 距离分辨率
        self.lambda_radar = self.c / startFreq  # 雷达信号波长
        self.isReal = isReal  # 1为实采样，0为复采样


def read_adc_data(filename, radar, process_num):
    """
    :param process_num: 帧的数量
    :return:
    """
    # Read Bin file
    with open(filename, 'rb') as f:
        adcDataRow = np.fromfile(f, dtype=np.int16)

    # 如果ADC采样位数不是16位，则需要进行转换
    if radar.numADCBits != 16:
        l_max = 2 ** (radar.numADCBits - 1) - 1
        adcDataRow[adcDataRow > l_max] -= 2 ** radar.numADCBits

    # 数据量大小：帧数*每帧中chirp数*每个chirp的采样点数*发射天线数*接收天线数*2（实部虚部）
    # 这里重新计算了fileSize，是为了了确保处理的数据块包含一个整数个脉冲重复次数（PRTs）
    fileSize = process_num * 2 * radar.numADCSamples * radar.numTX * radar.numRX * 2
    PRTnum = fileSize // (radar.numADCSamples * radar.numRX)
    fileSize = PRTnum * radar.numADCSamples * radar.numRX
    adcData = adcDataRow[:fileSize]

    if radar.isReal:
        totleChirps = fileSize // (radar.numADCSamples * radar.numRX)
        # Reshape the real data into a 2D array with each chirp as a column
        LVDS = np.reshape(adcData, (radar.numADCSamples * radar.numRX, totleChirps))
        # Transpose to make each row correspond to one chirp
        LVDS = LVDS.T
    else:
        # 计算总chirp数，含有实部虚部，故除以2
        totleChirps = fileSize // (2 * radar.numADCSamples * radar.numRX)
        # Pre-allocate array for complex data
        LVDS = np.zeros((1, fileSize // 2), dtype=complex)
        # 合并实部虚部
        counter = 0;
        for i in range(0, fileSize - 2, 4):
            LVDS[0, counter] = adcData[i] + 1j * adcData[i + 2]
            LVDS[0, counter + 1] = adcData[i + 1] + 1j * adcData[i + 3]
            counter += 2
        # Reshape to a 2D array with each chirp as a column
        LVDS = np.reshape(LVDS, (totleChirps, radar.numADCSamples * radar.numRX))

    # Reorganize data for each receiving antenna
    adcData_reorganized = np.zeros((radar.numRX, totleChirps * radar.numADCSamples), dtype=LVDS.dtype)
    for row in range(radar.numRX):
        for i in range(totleChirps):
            start_idx = i * radar.numADCSamples
            end_idx = start_idx + radar.numADCSamples
            adcData_reorganized[row, start_idx:end_idx] = LVDS[i,
                                                          row * radar.numADCSamples:(row + 1) * radar.numADCSamples]

    retVal = np.reshape(adcData_reorganized[0, :], (totleChirps, radar.numADCSamples)).T
    process_adc = np.zeros((radar.numADCSamples, totleChirps // 2), dtype=np.complex_)
    for nchirp in range(0, totleChirps, 2):
        process_adc[:, nchirp // 2] = retVal[:, nchirp]

    return process_adc, totleChirps


def smoothdata(signal, window_size):
    """模仿matlab中的smoothdata函数。 """
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


""" 1. 预制参数 & 读取原始数据"""
radar_params = Radar_params(numADCSamples=200, numADCBits=16,
                            numTX=1, numRX=4, chirpLoop=2,
                            Fs=4e6, slope=64.985e12, startFreq=60.25e9)
process_adc, totleChirps = read_adc_data(
    'data/adc_data22.bin', radar_params, 1024)

""" 2. 信号处理 """
# Perform FFT
fft1d = np.abs(fft(process_adc, axis=0)).T
X, Y = np.meshgrid(np.arange(radar_params.numADCSamples) * radar_params.detaR, np.arange(totleChirps // 2))

# 画距离 FFT 图, 20*log10() 意为将幅度转换为分贝
plt.figure()
plt.plot(np.arange(radar_params.numADCSamples) * radar_params.detaR, 20 * np.log10(fft1d[0, :]))
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

# 距离 FFT，注意转化成了1024 * 256的矩阵
fft_data = fft(adcdata, n=RangFFT, axis=0).T

# Calculate the magnitude (absolute value) of the complex FFT result
fft_data_abs = np.abs(fft_data)
"""
??? 为什么要调整距离分辨率
"""
# Adjust the distance resolution because the sample points have been expanded to 256
deltaR = radar_params.Fs * radar_params.c / (2 * radar_params.slope * RangFFT)
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
real_data = np.real(fft_data)  # R实部
imag_data = np.imag(fft_data)  # 虚部

# Calculate the phase for each range bin using atan2 for the entire matrix
angle_fft = np.arctan2(imag_data, real_data)

# Range-bin tracking - 找出能量最大的点，即人体的位置
range_max = 0
max_num = 0

# Initialize a storage array for energy accumulation
fft_data_last = np.zeros(RangFFT)

for j in range(RangFFT):
    # Check if the range bin is within the specified distance limits
    if (j * radar_params.detaR) < 2.5 and (j * radar_params.detaR) > 0.5:
        # Sum the energies for a non-coherent accumulation
        fft_data_last[j] = np.sum(fft_data_abs[:, j])

        # Update the maximum energy and corresponding bin number
        if fft_data_last[j] > range_max:
            range_max = fft_data_last[j]
            max_num = j

# Extract phase from the range bin with the maximum energy
angle_fft_last = angle_fft[:, max_num]

# Unwrap the phase
angle_fft_last = np.unwrap(angle_fft_last)

# Phase difference 相位差分
angle_fft_diff = np.diff(angle_fft_last)
angle_fft_diff = np.insert(angle_fft_diff, 0, 0)  # Inserting a zero at the beginning to maintain array size

# Moving average filtering

phi = smoothdata(angle_fft_diff, 5)

# FFT of the phase signal
N1 = len(phi)
FS = 20
fft_phase = np.abs(fft(phi))
f = np.arange(N1) * (FS / N1)

# Bandpass filter for breath signal
sosMatrix = np.array([[1, 0, -1, 1, -1.96315431309333, 0.964395116007807],
                      [1, 0, -1, 1, -1.85009994293252, 0.868089891124299]])
ScaleValues = np.array([0.0601804080654874, 0.0601804080654874, 1])
# 只对分子系数 (b) 应用缩放值，分母系数 (a) 保持不变
sosMatrix[:, :3] *= ScaleValues[0]  # 假设所有分子系数都应用同一个缩放值
breath_data = sosfilt(sosMatrix, phi)
breath_data *= ScaleValues[-1]

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

print(f"Breath Rate: {breath_count} breaths per minute")
plt.show()

""" 处理心率 """

# Bandpass filter for heart signal
sosMatrix = np.array(
    [[1, 0, -1, 1, -1.49069622543010, 0.821584856975103],
     [1, 0, -1, 1, -1.85299358095713, 0.916570170374599],
     [1, 0, -1, 1, -1.65884708697245, 0.751234896982056],
     [1, 0, -1, 1, -1.46163518810405, 0.655484575195733]])

ScaleValues = np.array([0.175431355008461, 0.175431355008461, 0.161866596066365, 0.161866596066365, 1])

# 只对分子系数 (b) 应用缩放值，分母系数 (a) 保持不变
sosMatrix[:, :3] *= ScaleValues[0]  # 假设所有分子系数都应用同一个缩放值
heart_data = sosfilt(sosMatrix, phi)
heart_data *= ScaleValues[-1]

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
heart_count = (fs * (N1 / 2 - (heart_index)) / N1) * 60

print(f"Heart Rate: {heart_count} beats per minute")

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
