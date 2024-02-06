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
        self.deltaR = self.c / (2 * self.B_valid)  # 距离分辨率
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

    # 重组数据retVal：200*2048矩阵，每一列为一个chirp
    # 取第一个接收天线数据
    retVal = np.reshape(adcData_reorganized[0, :], (totleChirps, radar.numADCSamples)).T
    # 每帧中的两个chrip取第一个，200*1024
    process_adc = np.zeros((radar.numADCSamples, totleChirps // 2), dtype=np.complex_)
    # 1T4R （1T1R）只处理单发单收的数据，并且只处理两个chrip取出的第一个
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


def sosfilter(signal, sosMatrix, ScaleValues):
    """模仿 matlab中的sosfilt函数。 """

    # 只对分子系数 (b) 应用缩放值，分母系数 (a) 保持不变
    sosMatrix[:, :3] *= ScaleValues[0]  # 假设所有分子系数都应用同一个缩放值
    signal = sosfilt(sosMatrix, signal)
    signal *= ScaleValues[-1]

    return signal

def easy_plot(x,y, title, xlabel, ylabel):
    plt.figure()
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
##
""" 1. 预制参数 & 读取原始数据"""
radar_para = Radar_params(numADCSamples=200, numADCBits=16,
                          numTX=1, numRX=4, chirpLoop=2,
                          Fs=4e6, slope=64.985e12, startFreq=60.25e9)
adc_data, totleChirps = read_adc_data('data/adc_data21.bin',
                                      radar_para, 1024)
##
""" 2. 信号处理 """

""" 2.1 距离 FFT （仅用于初步观察）"""
fft1d = np.abs(fft(adc_data, axis=0)).T


# 画距离 FFT 图, 20*log10() 意为将幅度转换为分贝
def plot_1d_fft(fft_signal, radar_params, idx=0):
    """    画某一个 chirp 的距离维 1D FFT 图，idx为 chirp 的编号    """
    plt.figure()
    plt.plot(np.arange(radar_params.numADCSamples) * radar_params.deltaR, 20 * np.log10(fft_signal[idx, :]))
    plt.xlabel('Range (m)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Range Dimension FFT of a single chirp')
    plt.show()


def plot_2d_fft(fft_signal, radar_params):
    """ 画出所有 chirp 的距离维 1D FFT, 形成 2D 图。该图也可以画成3D效果"""
    X, Y = np.meshgrid(np.arange(radar_params.numADCSamples) * radar_params.deltaR, np.arange(totleChirps // 2))
    plt.figure()
    plt.pcolormesh(X, Y, 20 * np.log10(fft_signal))
    plt.xlabel('Range (m)')
    plt.ylabel('Chirp number')
    plt.title('Range Dimension - 1D FFT Result')
    plt.colorbar()
    plt.show()


plot_1d_fft(fft1d, radar_para, idx=0)

plot_2d_fft(fft1d, radar_para)

##
""" 2.2 静态杂波消除 """

# 注意，此处的RangFFT是256，而不是原始的200，是为了其输入点数为2的幂次方。不是2的幂次方的点数会使FFT算法使用更慢的路径，增加计算时间。
RangFFT = 256
numChirps = adc_data.shape[1]

# 距离 FFT，但注意，由于RangeFFT是大于原始的200的，这里的fft()函数会自动补零。
# 此外，注意转化成了1024 * 256的矩阵
fft_data = fft(adc_data, n=RangFFT, axis=0).T
# 计算复数FFT结果的幅值（绝对值）
fft_data_abs = np.abs(fft_data)
# 设置完FFT的参数RangFFT之后，重新算下距离分辨率。
# 注意：不是雷达实际的距离分辨率，是最后成图时，range轴的间距为deltaR。
deltaR = radar_para.Fs * radar_para.c / (2 * radar_para.slope * RangFFT)
# 置零，去除直流分量，前10个已经基本可以帮助去除低频分量了
# 这里可以根据实际情况调整，例如，可以消除后面的背景波
fft_data_abs[:, :10] = 0


def plot_3d_fft(fft_signal, rangeFFT, deltaR, numChirps):
    """ 画出消除静态杂波后，所有 chirp 的距离维 1D FFT, 形成 3D 图。"""
    M, N = np.meshgrid(np.arange(rangeFFT) * deltaR, np.arange(1, numChirps + 1))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(M, N, fft_signal, cmap='viridis')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Chirp pulse number')
    ax.set_zlabel('Magnitude')
    ax.set_title('After Static Clutter Removal: Range-Dimension 1D FFT Result')
    plt.show()


plot_3d_fft(fft_data_abs, RangFFT, deltaR, numChirps)

##
""" 2.3 提取相位（相位反正切） & 目标距离识别 """

# 利用实部、虚部及反正切函数，计算相位
angle_fft = np.arctan2(np.real(fft_data), np.imag(fft_data))

# 找出能量最大的点，即人体的位置
fft_data_last = np.zeros(RangFFT)
detection_range = [0.5, 2.5]  # 检测0.5-2.5范围内的目标
for j in range(RangFFT):
    # 只对检测范围内的数据进行处理
    distance = j * radar_para.deltaR
    if detection_range[0] < distance < detection_range[1]:
        # Sum the energies for a non-coherent accumulation
        fft_data_last[j] = np.sum(fft_data_abs[:, j])

# Update the maximum energy and corresponding bin number
range_max, max_num = np.max(fft_data_last), np.argmax(fft_data_last)
##
""" 2.4 目标位置处的相位提取、解缠绕、差分、及平滑处理 """
# 提取能量最大的range bin的相位
angle_fft_last = angle_fft[:, max_num]
# 相位解缠绕，统一到 [-pi, pi]
angle_fft_last = np.unwrap(angle_fft_last)

# 相位差分，通过减去连续的相位值，对展开的相位执行相位差运算，
# 这将有利于：增强心跳信号并消除硬件接收机存在的相位漂移，抑制呼吸信号及其谐波
# 相位漂移是一个缓慢变化的过程，会影响接收到的信号的相位，但不会显著改变频率或相位变化的模式。
angle_fft_diff = np.diff(angle_fft_last)
# 由于差分会减少一个元素，所以需要在头部插入一个0。
angle_fft_diff = np.insert(angle_fft_diff, 0, 0)
# 窗口选择5，是根据实际经验确定的
phi = smoothdata(angle_fft_diff, 5)

##
""" 2.5 体征分析 """

""" 2.5.1 准备工作"""
# FFT of the phase signal
N1 = len(phi)
fs = 1024/51.2
fft_phase = np.abs(fft(phi))
f = np.arange(N1) * (fs / N1)  # Frequency range for each point in the FFT

""" 2.5.2 呼吸率计算 """
# 滤波器参数——网上下载的
# 注意，该sosMatrix中已经包含了截止频率、带宽、类型（低通、高通、带通、带阻）等信息，因此在sosfilter函数中不需要再传入这些参数。
sosMatrix = np.array([[1, 0, -1, 1, -1.96315431309333, 0.964395116007807],
                      [1, 0, -1, 1, -1.85009994293252, 0.868089891124299]])
ScaleValues = np.array([0.0601804080654874, 0.0601804080654874, 1])
breath_data = sosfilter(phi, sosMatrix, ScaleValues)


fshift = np.arange(-N1 / 2, N1 / 2) * (fs / N1)  # Zero-centered frequency range for plotting

# Perform FFT and shift it to center zero frequency
# fftshift 函数对 FFT 的输出进行重新排序，以便直流分量（零频率分量）位于中心。在 FFT 的标准输出中，零频率分量通常在数组的开始处，负频率分量跟在正频率分量之后。使用 fftshift 可以让负频率分量移到数组的前半部分，而正频率分量移到后半部分，使得频谱以零频率为中心对称排列。
breath_fre = np.abs(fftshift(fft(breath_data)))

# Search for the peak in the spectrum to determine the breath rate
breath_fre_max = np.max(breath_fre)
breath_index = np.argmax(breath_fre)

# Calculate breath rate
breath_count = (fs * (N1 / 2 - (breath_index)) / N1) * 60

print(f"Breath Rate: {breath_count} breaths per minute")
easy_plot(fshift[:N1 // 2], breath_fre[:N1 // 2],
          'Breath Signal FFT',
          'Frequency (f/Hz)',
          'Magnitude')
##
""" 2.5.3 心率计算 """

# 滤波器参数——网上下载的
sosMatrix = np.array(
    [[1, 0, -1, 1, -1.49069622543010, 0.821584856975103],
     [1, 0, -1, 1, -1.85299358095713, 0.916570170374599],
     [1, 0, -1, 1, -1.65884708697245, 0.751234896982056],
     [1, 0, -1, 1, -1.46163518810405, 0.655484575195733]])
ScaleValues = np.array([0.175431355008461, 0.175431355008461, 0.161866596066365, 0.161866596066365, 1])
heart_data = sosfilter(phi, sosMatrix, ScaleValues)

# Heart rate signal processing
fshift = np.arange(-N1 / 2, N1 / 2) * (fs / N1)  # Zero-centered frequency range

# Perform FFT and shift
heart_fre = np.abs(fftshift(fft(heart_data)))  # Double-sided spectrum (magnitude)
heart = np.abs(fft(heart_data))  # Single-sided spectrum (magnitude)

# Search for the maximum in the first half of the spectrum
heart_fre_max = np.max(heart_fre[:N1 // 2])
heart_index = np.argmax(heart_fre[:N1 // 2])

# Determine if a heartbeat is present based on magnitude confidence
if heart_fre_max < 1e-2:
    heart_index = N1  # Set to an invalid index if no heartbeat is detected

# Calculate the heart rate
heart_count = (fs * (N1 / 2 - (heart_index)) / N1) * 60

print(f"Heart Rate: {heart_count} beats per minute")

easy_plot(f[:200],heart[:200],
          'Heart Rate Signal FFT',
          'Frequency (f/Hz)',
          'Magnitude')


# Extract envelope for heart signal
analytic_signal = hilbert(heart_data)
env_envelope = np.abs(analytic_signal)

# Plotting the results
easy_plot(np.arange(N1), env_envelope,
          'Heartbeat Envelope',
          'Time (s)',
          'Amplitude')

# Normalizing heartbeat signal
normalized_heartbeat = heart_data / env_envelope
easy_plot(np.arange(N1), normalized_heartbeat,
          'Normalized Heartbeat Signal',
          'Time (s)',
          'Normalized Amplitude')