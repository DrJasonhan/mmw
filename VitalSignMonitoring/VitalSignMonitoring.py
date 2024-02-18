"""
代码内容:
1. 该代码是基于TI mmWave雷达的体征监测的matlab示例代码的python实现。
2. 适用的雷达型号包括： xWR16xx, IWR6843, xWR12xx, xWR14xx
3. 数据采集板卡：DCA1000EVM
4. 默认采集的信号都是复数形式
5. 该代码的实现流程包括：数据读取、目标距离识别、相位提取、相位解缠绕、相位差分、平滑处理、带通滤波、傅里叶变换。


Author: Shuai HAN
Email: shuaihan@polyu.edu.hk
Date: 2024.2.5
"""
##
import numpy as np
from scipy.fft import fft
from scipy.signal import hilbert, butter, filtfilt
from biosppy.signals import resp
import matplotlib.pyplot as plt
import seaborn as sns


class Radar_params:
    def __init__(self, numADCSamples, numADCBits, numTX, numRX, chirpLoop, Fs, slope, startFreq, numLanes, model):
        """
        :param model: 雷达的型号，
            0: xWR16xx and IWR6843【matlab demo中的代码】；
            1: xWR12xx and xWR14xx【我们的设备是1443】
        """
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
        self.numLanes = numLanes  # 通道数
        #
        self.model = model


class experiment_setup:
    def __init__(self, det_range0, det_range1, duration, process_num, filter_window):
        self.det_range0 = det_range0  # 感知范围 单位m
        self.det_range1 = det_range1  # 感知范围 单位m
        self.duration = duration  # 测量时间 s
        self.process_num = process_num  # 帧的数量
        self.filter_window = filter_window  # 滤波窗口大小


def read_adc_data(filename, radar, process_num):
    """ process_num: 一个Rx所对应的帧的数量    """
    # Read Bin file
    with open(filename, 'rb') as f:
        adcDataRow = np.fromfile(f, dtype=np.int16)

    # 如果ADC采样位数不是16位，则需要进行转换
    if radar.numADCBits != 16:
        l_max = 2 ** (radar.numADCBits - 1) - 1
        adcDataRow[adcDataRow > l_max] -= 2 ** radar.numADCBits

    # 数据量大小：帧数*每帧中chirp数*每个chirp的采样点数*发射天线数*接收天线数*2（实部虚部）
    # 这里重新计算了fileSize，是为了了确保处理的数据块包含一个整数个脉冲重复次数（PRTs）
    fileSize = process_num * radar.chirpLoop * radar.numADCSamples * radar.numTX * radar.numRX * 2
    PRTnum = fileSize // (radar.numADCSamples * radar.numRX)
    fileSize = PRTnum * radar.numADCSamples * radar.numRX
    adcData = adcDataRow[:fileSize]
    # 默认都是复数
    # 计算总chirp数，含有实部虚部，故除以2
    totalChirps = fileSize // (2 * radar.numADCSamples * radar.numRX * radar.numTX)

    # 对实部和虚部进行拼接
    adcData = adcData.reshape((radar.numLanes * 2, -1), order='F')
    adcData = (adcData[np.arange(0, radar.numLanes), :] +
               1j * adcData[np.arange(radar.numLanes, radar.numLanes * 2), :])

    """
    重排数据，并取第一个接收天线Rx数据（单发单收）。注意：
    matlab示例代码中，数据是按照Rx的顺序储存的，即前四分之一的数据全部是Rx0，然后是Rx1，Rx2，Rx3。
    我们雷达的数据排列方式为：Rx0，Rx1，Rx2，Rx3，Rx0，Rx1，Rx2，Rx3，...。
    
    规定 mode == 0 是matlab示例数据的排列方式，mode == 1 是我们雷达的数据排列方式。 
    """
    Rx_id = 0  #
    if radar.model == 0:
        adcData = adcData.flatten(order='F').reshape((
            totalChirps, radar.numADCSamples * radar.numRX))
        sigle_rx = adcData[:, radar.numADCSamples * Rx_id:radar.numADCSamples * (Rx_id + 1)]
    elif radar.model == 1:
        sigle_rx = adcData[Rx_id, :].reshape((totalChirps, -1))

    """
    对于上面Rx的接收天线，只选取每一个frame中的第一个chirp。原因是：
        1. 只取一个chirp的已满足了体征监测的频率需求；
        2. frame之间有时间差，若使用所有chirp，chirp间的时间间隔是不均等的，导致后续会有误差。
    """
    process_adc = sigle_rx[np.arange(0, totalChirps, radar.chirpLoop), :].T

    return process_adc, totalChirps


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


def plot_2d_fft(fft_signal, radar_params):
    """ 画出所有 chirp 的距离维 1D FFT, 形成 2D 图。该图也可以画成3D效果"""
    X, Y = np.meshgrid(np.arange(radar_params.numADCSamples) * radar_params.deltaR,
                       np.arange(totalChirps // radar_params.chirpLoop))
    plt.figure()
    plt.pcolormesh(X, Y, 20 * np.log10(fft_signal))
    plt.xlabel('Range (m)')
    plt.ylabel('Chirp number')
    plt.title('Range Dimension - 1D FFT Result')
    plt.colorbar()
    plt.show()


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


##
""" 1. 预制参数 & 读取原始数据"""
# Matlab demo 的配置
the_radar = Radar_params(numADCSamples=200, numADCBits=16,
                         numTX=1, numRX=4, chirpLoop=2,
                         Fs=4e6, slope=64.985e12, startFreq=60.25e9, numLanes=2, model=0)
the_setup = experiment_setup(det_range0=0.5, det_range1=2.5,
                             duration=51.2, process_num=1024, filter_window=5)
data_path = 'data/adc_data21.bin'

# 我们设备的配置
# the_radar = Radar_params(numADCSamples=256, numADCBits=16,
#                          numTX=1, numRX=4, chirpLoop=4,
#                          Fs=1e7, slope=29.98e12, startFreq=77e9, numLanes=4, model=1)
# the_setup = experiment_setup(det_range0=0.2, det_range1=2.5,
#                              duration=40, process_num=40 * 25, filter_window=6)
# data_path = 'data/adc_data_test.bin'
adc_data, totalChirps = read_adc_data(data_path, the_radar, the_setup.process_num)
##
""" 2. 信号处理 """

""" 2.1 距离 FFT （仅用于初步观察）"""
fft1d = np.abs(fft(adc_data, axis=0)).T

sns.lineplot(x=np.arange(the_radar.numADCSamples) * the_radar.deltaR,
             y=20 * np.log10(fft1d[1, :])).set(
    title='MRange Dimension FFT of a single chirp',
    xlabel='Range (m)', ylabel='Magnitude (dB)')

plot_2d_fft(fft1d, the_radar)

##
""" 2.2 静态杂波消除 """

# 注意，此处的RangFFT不一定与numADCSamples一致，而是调整为≥numADCSamples的、最小的2的幂次方。
# 不是2的幂次方的点数会使FFT算法使用更慢的路径，增加计算时间。
RangFFT = 2 ** np.ceil(np.log2(the_radar.numADCSamples)).astype(int)

# 注意，转置了
fft_data = fft(adc_data, n=RangFFT, axis=0).T
# 计算复数FFT结果的幅值（绝对值）
fft_data_abs = np.abs(fft_data)
# 设置完FFT的参数RangFFT之后，重新算下距离分辨率。
# 注意：不是雷达实际的距离分辨率，是最后成图时，range轴的间距为deltaR。
deltaR = the_radar.Fs * the_radar.c / (2 * the_radar.slope * RangFFT)
# 置零，去除直流分量，前10个已经基本可以帮助去除低频分量了
# 这里可以根据实际情况调整，例如，可以消除身后墙壁反射的背景波
fft_data_abs[:, :10] = 0

plot_3d_fft(fft_data_abs, RangFFT, deltaR, adc_data.shape[1])

##
""" 2.3 提取相位（相位反正切） & 目标距离识别 """

# 找出能量最大的点，即人体的位置
slice0, slice1 = int(the_setup.det_range0 // deltaR), int(the_setup.det_range1 // deltaR + 1)
# 沿着chirp轴求和 切片保留需要的距离bin
power_sum = np.sum(fft_data_abs[:, slice0:slice1], axis=0)
range_max, max_num = np.max(power_sum), np.argmax(power_sum) + slice0

##
""" 2.4 目标位置处的相位提取、解缠绕、差分、及平滑处理 """
# 使用np.angle函数，原理是利用实部、虚部计算反正切，进而得到相位
angle_fft = np.angle(fft_data)  # return rad
# 提取能量最大的range bin的相位
angle_fft_human = angle_fft[:, max_num]
# 相位解缠绕，统一到 [-pi, pi]
angle_fft_human = np.unwrap(angle_fft_human)
"""
相位差分，通过减去连续的相位值，对展开的相位执行相位差运算，
目的是增强心跳信号并消除硬件接收机存在的相位漂移，抑制呼吸信号及其谐波
相位漂移是一个缓慢变化的过程，会影响接收到的信号的相位，但不会显著改变频率或相位变化的模式。
"""
angle_fft_diff = np.diff(angle_fft_human)
# 重要的事情：由于差分会减少一个元素，所以需要在头部插入一个0。
angle_fft_diff = np.insert(angle_fft_diff, 0, 0)

##
""" 2.5 体征分析 """

""" 2.5.1 准备工作"""
# 默认选取 0.25 s 的滑动窗口，窗口长度为5
phi = smoothdata(angle_fft_diff, the_setup.filter_window)

N = len(phi)
fs = the_setup.process_num / the_setup.duration  # 采样频率
f = np.arange(N) * (fs / N)  # FFT 后的频率轴

##
""" 2.5.2 呼吸率计算 """
# resp()函数默认滤波范围是0.1-0.35hz，FIR
bd = resp.resp(signal=phi, sampling_rate=fs, show=True)
breath_count = np.mean(bd[4]) * 60
print(f"Breath Rate: {breath_count: .2f} breaths per minute")

##
""" 2.5.3 心率计算 """
# 滤波
b, a = butter(2, [0.67, 2.5], btype='bandpass', fs=fs)
heart_data = filtfilt(b, a, phi)

# 傅里叶变换，因为计算出来是双边谱，所以只取一半
heart = np.abs(fft(heart_data))[:N // 2]

# 选取幅值最大的频率，即心率
heart_fre_max = np.max(heart)
heart_index = np.argmax(heart)

# 根据振幅判断是否有心跳
if heart_fre_max < 1e-2:
    print("没有检测到心跳信号")

# 频率-->心率
heart_count = f[heart_index] * 60

print(f"Heart Rate: {heart_count: .2f} beats per minute")

# 画图：
plt.figure()
sns.lineplot(x=f[:N // 2], y=heart).set(
    title='Heart Rate Signal FFT',
    xlabel='Frequency (f/Hz)',
    ylabel='Magnitude')

# Extract envelope for heart signal
analytic_signal = hilbert(heart_data)
env_envelope = np.abs(analytic_signal)
plt.figure()
sns.lineplot(x=np.arange(N), y=heart_data).set(
    title='Heartbeat Signal',
    xlabel='Time (s)',
    ylabel='Amplitude')
sns.lineplot(x=np.arange(N), y=env_envelope)

# 心跳信号归一化
normalized_heartbeat = heart_data / env_envelope
plt.figure()
sns.lineplot(x=np.arange(N), y=normalized_heartbeat).set(
    title='Normalized Heartbeat Signal',
    xlabel='Time (s)',
    ylabel='Normalized Amplitude')
