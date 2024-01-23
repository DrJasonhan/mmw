import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from scipy import signal
import scipy.io as sio

import heartpy as hp

import matplotlib.pyplot as plt





# Radar 的配置参数
num_samples = 256
Fs = 5e6
slope = 70.006e12
t_frame = 0.01
cpf = 16


numFrames = 4000  # 帧数

# 呼吸的参数
br_filt_order = 2  # 滤波阶数
br_lower_cut = 0.2  # 截止频率
br_upper_cut = 0.6

# 心跳的参数
hr_filt_order = 2  # 滤波阶数
hr_lower_cut = 1  # 截止频率
hr_upper_cut = 2.5


# 设备参数                   #描述 单位
c = 3e8  # 光速 m/s
ts = num_samples / Fs  # ADC采样时间 s
B_valid = ts * slope  # 有效带宽：Hz
detaR = c / (2 * B_valid)  # 距离分辨率：m
# fs_slow = cpf / t_frame  # chirp的采样频率

# adc = np.fromfile("Page/LaboratoryFunc/mmwVitalSign.bin", dtype=np.int16)
mat_data = sio.loadmat('data/test_wl70summer.mat')  # 从MAT文件读取数据
DataOneChirpMultiFrame = mat_data['DataOneChirpMultiFrame_transpose']  # 获取数据
# 如果数据帧数不足，使用零填充至指定的帧数
data = np.concatenate((
    DataOneChirpMultiFrame, np.zeros((numFrames - DataOneChirpMultiFrame.shape[0], num_samples))),
    axis=0).T


# # model iwr1443 boost
# # 1000帧 每帧4chirp 每个chirp256个样本 数据为int16复数 每帧40ms
# adc_data = readDCA1000(adc, isReal=False)  # (4, num_samples x num_chirps)
# data = adc_data[0, :].reshape(num_samples, -1, order='F')  # 只处理一个天线的数据 (num_samples, num_chirps)




"""# FFT over fast time"""
fft_data = fft(data, axis=0)  #

fft_y = fft_data[:, 7]  # 展示一个chirp的频谱
abs_y = np.abs(fft_y) / num_samples  # 取复数模长，然后归一化
abs_y = abs_y[:num_samples // 2]  # 取一半
xf = fftfreq(num_samples, 1 / Fs)[:num_samples // 2]

fft_data = fft_data.T  # (num_chirp, num_sample)
fft_data_abs = np.abs(fft_data)

"""# 提取相位"""
angle_fft = np.angle(fft_data)  # return rad

# 找到能量最大的距离bin 即人的位置
det_range0 = 0.2
det_range1 = 4  # 查找范围 单位m
slice0 = int(det_range0 // detaR)
slice1 = int(det_range1 // detaR + 1)
# 沿着chirp轴求和 切片保留需要的距离bin
power_sum = np.sum(fft_data_abs[:, slice0:slice1], axis=0)
range_max = np.argmax(power_sum)


# 取出人所在位置的这一列的相位
angle_fft_last = angle_fft[:, slice0:slice1][:, range_max]
# angle_fft_last = angle_fft[:,10]

"""# 相位解缠绕"""
# 每当连续值之间的相位差大于或者小于±π时，通过从相位中减去2π来获得相位展开
phi = angle_fft_last  # (n_chirps,)
assert angle_fft_last.shape.__len__() == 1
angle_fft_last = np.unwrap(phi)

# 相位差分
angle_fft_last2 = np.diff(angle_fft_last)  # (n_chirps-1,)

"""滑动平均滤波"""
window_size = 16  # chirp数量
kernel = np.ones(window_size) / window_size
phi = np.convolve(angle_fft_last2, kernel, mode='same')  # 利用一维卷积实现滤波

"""#  IIR带通滤波 Bandpass Filter 0.1-0.5hz，输出呼吸信号"""
#  构造butterworth滤波参数
b, a = signal.butter(br_filt_order, [br_lower_cut, br_upper_cut], btype='bandpass', fs=1/t_frame)
breath_data = signal.filtfilt(b, a, phi)

"""#  IIR带通滤波 Bandpass Filter 0.8-2hz，输出心跳信号"""
#  构造butterworth滤波参数
b, a = signal.butter(hr_filt_order, [hr_lower_cut, hr_upper_cut], btype='bandpass', fs=1/t_frame)
heart_data = signal.filtfilt(b, a, phi)

# 求心跳波的包络
window_size = 16  # chirp数量
kernel = np.ones(window_size) / window_size
env = np.convolve(np.abs(heart_data), kernel, mode='same')

# 心跳再次滤波
window_size = 16  # chirp数量
kernel = np.ones(window_size) / window_size
heart_data_ = np.convolve(heart_data, kernel, mode='same')

filtered = hp.filter_signal(angle_fft_last2, cutoff=[hr_lower_cut, hr_upper_cut], sample_rate=24.95,
                            order=2, filtertype='bandpass')
working_data, measures = hp.process(filtered, 24.95)
heartbeat_rate = np.int32(measures['bpm'])

"""
！！ 注意：当前的呼吸时利用心率推断出来的，而非直接通过毫米波数据测出的。后期需要修改。
原理：
    心跳和呼吸之间可能存在一定的相关性，这种现象被称为————心呼吸耦合。
    例如，深度放松或在进行某些呼吸练习时，人的心跳和呼吸可能会同步化。
    在这些情况下，通过心跳变异性（HRV）分析，一种衡量心率变化的方法，可能间接推断出呼吸模式的变化。
    HRV中的某些参数，如呼吸波（respiratory sinus arrhythmia, RSA），与呼吸节律有关，可以提供呼吸频率的间接信息。
"""
breath_rate = np.int32(measures["breathingrate"]*60)

print("Heart beat: {0}; Breath rate: {1}".format(heartbeat_rate,breath_rate))
plt.plot(breath_data)
plt.show()