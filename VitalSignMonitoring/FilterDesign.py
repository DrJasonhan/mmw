import numpy as np
from scipy.signal import butter, buttord, sosfreqz

import matplotlib.pyplot as plt

wp = [0.2, 0.5]  # 通带边界频率
ws = [0.15, 0.55]  # 阻带边界频率

# 定义在通带和阻带中允许的最大损耗（以分贝为单位）
gpass = 1  # 通带最大损耗
gstop = 30  # 阻带最大损耗
# 采样频率
fs = 25

# 计算滤波器的最小阶数和截止频率
N, Wn = buttord(wp, ws, gpass, gstop, fs=fs)
sos = butter(N, Wn, 'band', fs=fs, output='sos')

# 计算频率响应
w, h = sosfreqz(sos, fs=fs)

# 计算幅度响应（以分贝为单位）
amp = 20 * np.log10(np.abs(h))

# 绘制幅度响应曲线
plt.plot(w, amp)
plt.title('Frequency response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.grid(True)
plt.show()

# t = np.linspace(0, 1, 500, False)  # 1 second
# sig1 = np.sin(2*np.pi*5*t)  # 5 Hz signal
# sig2 = 0.5 * np.sin(2*np.pi*10*t)  # 10 Hz signal
# sig3 = 0.3 * np.sin(2*np.pi*20*t)  # 20 Hz signal
# noise = np.random.normal(size=500)*0.2  # Random noise
# signal = sig1 + sig2 + sig3 + noise  # Complex signal
#
# 使用 SOS 滤波器对信号进行滤波
# filtered_signal = sosfiltfilt(sos, signal)
# plt.plot(signal)
# plt.plot(filtered_signal)
