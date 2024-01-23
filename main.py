"""
将任博士的matlab代码转化为python代码
"""
import numpy as np
import scipy.io as sio
from scipy.signal import remez, lfilter, firwin, firls

import matplotlib

# 设置matplotlib的backend为TkAgg，这样可以在多数环境中显示图形界面
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 设置Matplotlib支持中文字符
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# 定义处理参数的类
class ProcPara():
    """ 处理参数类 """

    def __init__(self):
        self.FlagRangeCellSelection = 0  # 距离单元选择标志（0表示选择最大功率单元）
        self.FlagProcMethod = 1  # 处理方法标志
        self.numFrames = 4000  # 帧数
        self.RangeOfInterest = [0.3, 4]  # 兴趣范围(m)


# 定义雷达参数的类
class RadarPara():
    def __init__(self):
        self.numADCSamples = 256  # ADC样本数
        self.numADCBits = 16  # ADC位数
        self.numTx = 3  # 发射器数量
        self.numRx = 4  # 接收器数量
        self.numLanes = 2  # 数据通道数量
        self.isReal = 0  # 是否是实数
        self.c = 3e8  # 光速(m/s)
        self.fc = 77e9  # 载频(Hz)
        """
        KChirp 线性调频(Linear Frequency Modulation, LFM) 信号的调频斜率，即频率随时间变化的速率。较大的斜率可产生较短的压缩脉冲宽度，从而获得较高的距离分辨率。
        """
        self.KChirp = 70.006e12  # 线性调频斜率(Hz/s)
        self.TIdleTime = 100e-6  # 空闲时间(s)
        self.TRampEndTime = 56.87e-6  # 斜坡结束时间(s)

        self.fs = 5e6  # 采样频率(Hz)
        self.PeriodFrame = 10e-3  # 帧周期(s)
        self.numChirpLoops = 16  # 每帧chirp循环数
        """
        PRT(Pulse Repetition Time) ,连续两个发射脉冲之间的时间间隔。
        """
        self.PRT = self.TIdleTime + self.TRampEndTime  # 脉冲重复时间(s)
        self.B = self.KChirp * self.numADCSamples / self.fs  # 带宽(Hz)
        self.Rres = self.c / 2 / self.B  # 距离分辨率(m)
        self.Vmax = self.c / 4 / self.PRT / self.fc  # 最大速度(m/s)
        self.Vres = self.c / 2 / self.PRT / self.fc / self.numChirpLoops  # 速度分辨率(m/s)


# 创建处理参数和雷达参数的实例
proc = ProcPara()
radar = RadarPara()

# 加载数据
mat_data = sio.loadmat('data/test_wl70summer.mat')  # 从MAT文件读取数据
DataOneChirpMultiFrame = mat_data['DataOneChirpMultiFrame_transpose']  # 获取数据
# 如果数据帧数不足，使用零填充至指定的帧数
DataOneChirpMultiFrame = np.concatenate((
    DataOneChirpMultiFrame, np.zeros((proc.numFrames - DataOneChirpMultiFrame.shape[0], radar.numADCSamples))),
    axis=0).T

# 根据FlagRangeCellSelection标志选择目标位置
# 计算每个距离单元在所有帧中的总功率
totalPowerMultiFrame = np.sum(np.abs(DataOneChirpMultiFrame) ** 2, axis=1)
# 创建距离轴
Raxis = np.arange(0, radar.numADCSamples) * radar.Rres
# 限定兴趣范围内的总功率
totalPowerMultiFrameOfInterest = totalPowerMultiFrame[(Raxis <= proc.RangeOfInterest[1])]
# 找到功率最大的距离单元
IndexRowTarget = np.argmax(totalPowerMultiFrameOfInterest)
# IndexRowTarget =19
SeqOfRangeCell = 0
# 获取该距离单元的数据
dataOfInterestMultiFrame = DataOneChirpMultiFrame[IndexRowTarget + SeqOfRangeCell, :]


# 定义带通滤波器函数
def filter_bandpass(numtaps, low_cutoff, high_cutoff, signal):
    """
    使用remez算法设计FIR带通滤波器并应用于信号。

    参数:
    numtaps: 滤波器阶数。
    low_cutoff: 低截止频率（归一化到奈奎斯特频率）。
    high_cutoff: 高截止频率（归一化到奈奎斯特频率）。
    signal: 输入信号。

    返回:
    filtered_signal: 应用带通滤波器后的信号。
    """
    # 使用Remez算法设计FIR带通滤波器
    # 注意：remez函数中的频率向量需要归一化到[0, 0.5]，因为1对应奈奎斯特频率
    coefficients = remez(numtaps,
                         [0, 0.999 * low_cutoff, low_cutoff, high_cutoff, high_cutoff * 1.0001, 1],
                         [0, 1, 0],  # 增益向量，对应于阻带、通带、阻带
                         [1, 1, 1])  # 权重向量，通常设置为相等权重
    # 使用设计的FIR滤波器滤波信号
    filtered_signal = lfilter(coefficients, [1], signal)

    return filtered_signal


def filter_bandpass_original(numtaps, low_cutoff, high_cutoff, signal, fs):
    """
    使用firls算法设计FIR带通滤波器并应用于信号。

    参数:
    numtaps: 滤波器阶数。
    low_cutoff: 低截止频率（物理频率）。
    high_cutoff: 高截止频率（物理频率）。
    signal: 输入信号。
    fs: 采样频率。

    返回:
    filtered_signal: 应用带通滤波器后的信号。
    """
    # 归一化截止频率
    nyquist = fs / 2
    f1 = low_cutoff / nyquist
    f2 = high_cutoff / nyquist

    # 设计FIR带通滤波器
    coefficients = remez(numtaps,
                         [0, 0.999 * f1, f1, f2, f2 * 1.0001, 1],
                         [0, 1, 0],  # 增益向量，对应于阻带、通带、阻带
                         [1, 1, 1])  # 权重向量，通常设置为相等权重
    # 使用设计的FIR滤波器滤波信号
    filtered_signal = lfilter(coefficients, [1], signal)

    return filtered_signal


# 相位提取，对感兴趣数据的相位进行解包
SigUnwrapped = np.unwrap(np.angle(dataOfInterestMultiFrame))

"""
以上步骤，和matlab结果完全一致
——20240123
"""
# 设计FIR带通滤波器，并应用于解包后的相位信号，以提取呼吸相关信号
# b = firwin(51, [0.2 / 100, 0.6 / 100], pass_zero=False)
freq_measure = radar.numChirpLoops / radar.PeriodFrame
SigFiltered_RR = filter_bandpass(51, 0.2, 0.6, SigUnwrapped, 100)

# 心跳信号处理，设计带通滤波器，并应用于解包后的相位信号，以提取心跳相关信号
low_hb = 1  # 心跳信号低截止频率
high_hb = 2.5  # 心跳信号高截止频率
SigFiltered_HB = filter_bandpass(51, low_hb, high_hb, SigUnwrapped, 100)

# 绘制呼吸
dataOfInterest = SigFiltered_RR
Xplot = np.arange(0, proc.numFrames) * radar.PeriodFrame
Yplot = dataOfInterest

# Plot the respiration signal
plt.figure(figsize=(10, 8))
plt.plot(Xplot, Yplot, color='r', linestyle='-', linewidth=2)
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Phase (rad)')
plt.show()

# Plot the respiration signal with Chinese labels
plt.figure(figsize=(10, 8))
plt.plot(Xplot, Yplot, color='k', linestyle='-', linewidth=2)
plt.grid(True)
plt.xlabel('时间/s')
plt.ylabel('相位/rad')
plt.show()

# Plot, spectrum
Yplotfft = np.fft.fft(dataOfInterest)
Yplotfftdb = 20 * np.log10(np.abs(Yplotfft) / np.max(np.abs(Yplotfft)))
Xaxis = np.arange(0, proc.numFrames) / proc.numFrames / radar.PeriodFrame

# Plot the spectrum of the respiration signal
plt.figure(figsize=(10, 8))
plt.plot(Xaxis, Yplotfftdb, color='r', linestyle='-', linewidth=2)
plt.grid(True)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.xlim([0, 2])
plt.ylim([-40, None])
plt.show()

# Plot the spectrum of the respiration signal with Chinese labels
plt.figure(figsize=(10, 8))
plt.plot(Xaxis, Yplotfftdb, color='k', linestyle='-', linewidth=2)
plt.grid(True)
plt.xlabel('频率/Hz')
plt.ylabel('幅度/dB')
plt.xlim([0, 2])
plt.ylim([-40, None])
plt.show()

# Find the respiration rate
freq_range_id = (Xaxis >= 0.2) & (Xaxis <= 0.6)
freq_range_db = Yplotfftdb[freq_range_id]
max_db_id = np.argmax(freq_range_db)
max_db_id += np.where(freq_range_id)[0][0]
RR = Xaxis[max_db_id] * 60
print(f'The respiration rate is {RR} bpm')

# 绘制心跳
dataOfInterest = SigFiltered_HB
Xplot = np.arange(0, proc.numFrames) * radar.PeriodFrame
Yplot = dataOfInterest

# Plot the heartbeat signal
plt.figure(figsize=(10, 8))
plt.plot(Xplot, Yplot, color='r', linestyle='-', linewidth=2)
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Phase (rad)')
plt.show()

# Plot the heartbeat signal with Chinese labels
plt.figure(figsize=(10, 8))
plt.plot(Xplot, Yplot, color='k', linestyle='-', linewidth=2)
plt.grid(True)
plt.xlabel('时间/s')
plt.ylabel('相位/rad')
plt.show()

# Plot, spectrum
Yplotfft = np.fft.fft(dataOfInterest)
Yplotfftdb = 20 * np.log10(np.abs(Yplotfft) / np.max(np.abs(Yplotfft)))
Xaxis = np.arange(0, proc.numFrames) / proc.numFrames / radar.PeriodFrame

# Plot the spectrum of the heartbeat signal
plt.figure(figsize=(10, 8))
plt.plot(Xaxis, Yplotfftdb, color='r', linestyle='-', linewidth=2)
plt.grid(True)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.xlim([0, 2])
plt.show()

# Plot the spectrum of the heartbeat signal with Chinese labels
plt.figure(figsize=(10, 8))
plt.plot(Xaxis, Yplotfftdb, color='k', linestyle='-', linewidth=2)
plt.grid(True)
plt.xlabel('频率/Hz')
plt.ylabel('幅度/dB')
plt.xlim([0, 2])
plt.ylim([-40, None])
plt.show()

# Find the heartbeat rate
freq_range_id = (Xaxis >= low_hb) & (Xaxis <= high_hb)
freq_range_db = Yplotfftdb[freq_range_id]
max_db_id = np.argmax(freq_range_db)
max_db_id += np.where(freq_range_id)[0][0]
HR = Xaxis[max_db_id] * 60
print(f'The Heartbeat rate is {HR} bpm')
