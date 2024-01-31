import numpy as np
from scipy.fft import fft, fftfreq
from scipy import signal
import scipy.io as sio
import heartpy as hp
from biosppy.signals import resp, ecg
import matplotlib

matplotlib.use('Qt5Agg')
"""
model iwr1443 boost
1000帧 每帧4chirp 每个chirp256个样本 数据为int16复数 每帧40ms
"""

# Radar 配置
num_samples = 256
Fs = 5e6
slope = 70.006e12
t_frame = 0.01
cpf = 16  # chirp per frame
numFrames = 4000  # 帧数

# 设备参数
c = 3e8  # 光速 m/s
ts = num_samples / Fs  # ADC采样时间 s
B_valid = ts * slope  # 有效带宽：Hz
detaR = c / (2 * B_valid)  # 距离分辨率：m
fs_slow = cpf / t_frame  # chirp的采样频率

# adc = np.fromfile("Page/LaboratoryFunc/mmwVitalSign.bin", dtype=np.int16)
mat_data = sio.loadmat('data/0521_wl70summer-30.mat')  # 从MAT文件读取数据
data = mat_data['DataOneChirpMultiFrame_transpose']  # 获取数据
# 如果数据帧数不足，使用零填充至指定的帧数
data = np.concatenate((data, np.zeros((numFrames - data.shape[0], num_samples))),
                      axis=0).T

""" FFT over fast time"""
fft_data = fft(data, axis=0).T  # (num_chirp, num_sample)
fft_data_abs = np.abs(fft_data)

"""提取相位"""
angle_fft = np.angle(fft_data)  # return rad

# 找到能量最大的距离bin 即人的位置
det_range0, det_range1 = 0.2, 4 # 查找范围 单位m
slice0, slice1= int(det_range0 // detaR), int(det_range1 // detaR + 1)
# 沿着chirp轴求和 切片保留需要的距离bin
power_sum = np.sum(fft_data_abs[:, slice0:slice1], axis=0)
range_max = np.argmax(power_sum)

# 取出人所在位置的这一列的相位
angle_fft_last = angle_fft[:, slice0:slice1][:, range_max]
# angle_fft_last = angle_fft[:,10]

""" 相位解缠绕"""
# 每当连续值之间的相位差大于或者小于±π时，通过从相位中减去2π来获得相位展开
phi = angle_fft_last  # (n_chirps,)
angle_fft_last = np.unwrap(phi)

# 相位差分
angle_fft_last2 = np.diff(angle_fft_last)  # (n_chirps-1,)

"""滑动平均滤波"""
window_size = 16  # chirp数量
kernel = np.ones(window_size) / window_size
phi = np.convolve(angle_fft_last2, kernel, mode='same')  # 利用一维卷积实现滤波

""" 采用biosppy库中的resp.resp()函数实现呼吸滤波，其默认滤波范围是0.1-0.35hz，
注意：使用的滤波器为FIR"""
breath_rate = resp.resp(phi, 100)

"""  IIR带通滤波 Bandpass Filter 0.8-2hz，输出心跳信号"""
# 心跳的参数
hr_filt_order = 2  # 滤波阶数
hr_lower_cut, hr_upper_cut = 1, 2.5  # 截止频率

#  构造butterworth滤波参数
# b, a = signal.butter(hr_filt_order, [hr_lower_cut, hr_upper_cut],
#                      btype='bandpass', fs=1 / t_frame)
# heart_data = signal.filtfilt(b, a, phi)
#
#
#
# # 求心跳波的包络
# window_size = 16  # chirp数量
# kernel = np.ones(window_size) / window_size
# env = np.convolve(np.abs(heart_data), kernel, mode='same')
#
# # 心跳再次滤波
# window_size = 16  # chirp数量
# kernel = np.ones(window_size) / window_size
# heart_data_ = np.convolve(heart_data, kernel, mode='same')
# filtered = hp.filter_signal(angle_fft_last2, cutoff=[hr_lower_cut, hr_upper_cut], sample_rate=100,
#                             order=2, filtertype='bandpass')
# working_data, measures = hp.process(filtered, 100)
# heartbeat_rate = np.int32(measures['bpm'])

ht = ecg.ecg(phi, sampling_rate=100)
# working_data, measures = hp.process(phi, 100)

print("Heart beat: {0}; Breath rate: {1}".format(np.mean(ht[6]),
                                                 int(np.mean(breath_rate[4]) * 60)))
