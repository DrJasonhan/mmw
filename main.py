import numpy as np
from scipy.signal import firwin, filtfilt
import scipy.io as sio
import matplotlib
matplotlib.use('TkAgg') # Or any other backend
import matplotlib.pyplot as plt

from scipy.signal import fir_filter_design as design
from scipy.signal import lfilter

# Processing parameters
FlagRangeCellSelection = 0
FlagProcMethod = 1
numFrames = 4000
numTx = 3
KChirp = 70.006e12
TIdleTime = 100e-6
TRampEndTime = 56.87e-6
fs = 5e6
PeriodFrame = 10e-3
numChirpLoops = 16
RangeOfInterest = [0.3, 4]

# Parameters
numADCSamples = 256
numADCBits = 16
numRx = 4
numLanes = 2
isReal = 0
c = 3e8
fc = 77e9
PRT = TIdleTime + TRampEndTime
B = KChirp*numADCSamples/fs
Rres = c/2/B
Vmax = c/4/PRT/fc
Vres = c/2/PRT/fc/numChirpLoops


# 加载数据
mat_data = sio.loadmat('data/test.mat')
DataOneChirpMultiFrame = mat_data['DataOneChirpMultiFrame_transpose']
DataOneChirpMultiFrame = np.concatenate((
    DataOneChirpMultiFrame, np.zeros((numFrames-DataOneChirpMultiFrame.shape[0], numADCSamples))),
    axis=0).T



# 找到目标位置
if FlagRangeCellSelection == 0:
    totalPowerMultiFrame = np.sum(np.abs(DataOneChirpMultiFrame)**2, axis=1)
    Raxis = np.arange(0, numADCSamples)*Rres
    totalPowerMultiFrameOfInterest = totalPowerMultiFrame[(Raxis<=RangeOfInterest[1])]
    IndexRowTarget = np.argmax(totalPowerMultiFrameOfInterest)
    SeqOfRangeCell = 0
    dataOfInterestMultiFrame = DataOneChirpMultiFrame[IndexRowTarget+SeqOfRangeCell, :]
elif FlagRangeCellSelection == 1:
    dataOfInterestMultiFrame = np.zeros(numFrames)
    for idxFrame in range(numFrames):
        DataOneChirpOneFrame = DataOneChirpMultiFrame[:,idxFrame]
        IndexRowTarget = np.argmax(np.abs(DataOneChirpOneFrame))
        dataOfInterestMultiFrame[idxFrame] = DataOneChirpMultiFrame[IndexRowTarget, idxFrame]


def filter_bandpass(numtaps, low_cutoff, high_cutoff, signal):
    """
    Design a FIR bandpass filter using the remez algorithm and apply it to a signal.

    Parameters:
    numtaps: The filter order.
    low_cutoff: The low cutoff frequency (normalized to the Nyquist frequency).
    high_cutoff: The high cutoff frequency (normalized to the Nyquist frequency).
    signal: The input signal to be filtered.

    Returns:
    filtered_signal: The signal after being processed by the bandpass filter.
    """
    # Design the FIR bandpass filter using the Remez algorithm.
    coefficients = design.remez(numtaps,
                                [0, 0.999 * low_cutoff, low_cutoff, high_cutoff, high_cutoff * 1.0001, 0.5],
                                [0, 1, 0])

    # Use the designed FIR filter to filter the signal.
    filtered_signal = lfilter(coefficients, 1, signal)

    return filtered_signal


# 相位提取
SigUnwrapped = np.unwrap(np.angle(dataOfInterestMultiFrame))
b = firwin(51, [0.2 / 100, 0.6 / 100], pass_zero=False)
SigFiltered_RR = filter_bandpass(51, 0.2 / 100, 0.6 / 100, SigUnwrapped)

# 心跳信号处理
low_hb = 1
high_hb = 2.5
SigFiltered_HB = filter_bandpass(51, low_hb / 100, high_hb / 100, SigUnwrapped)

# 绘制呼吸信号

dataOfInterest = SigFiltered_RR
Xplot = np.arange(0, numFrames) * PeriodFrame
plt.figure()
plt.plot(Xplot, dataOfInterest)
plt.xlabel('Time (s)')
plt.ylabel('Phase (rad)')
# 其他设置


# 呼吸频谱分析
Yplotfft = np.fft.fft(dataOfInterest)
Yplotfftdb = 20*np.log10(np.abs(Yplotfft)/np.max(np.abs(Yplotfft)))
Xaxis = np.arange(0,numFrames)/numFrames * 1/PeriodFrame
plt.figure()
plt.plot(Xaxis, Yplotfftdb)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
# 找到峰值,计算呼吸率

# 绘制和分析心跳信号
# 方法同呼吸信号