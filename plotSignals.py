import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sosfiltfilt, butter

lc = np.load('local_3x3_long_hf.npy')
gb = np.load('global_3x3_long_hf.npy')
fc = np.load('global_5x5_long_hf.npy')
t = lc[0, :] * 10

sos = butter(8, 0.1, output='sos')
ls = sosfiltfilt(sos, lc[1, :])
gs = sosfiltfilt(sos, gb[1, :])
fs = sosfiltfilt(sos, fc[1, :])

plt.plot(t, ls, label='3x3 Local Convolution')
plt.plot(t, gs, label='3x3 Global Convolution')
plt.grid()
plt.legend()
plt.xlabel('Samples')
plt.ylabel("Error Rate")
plt.savefig("comparison_full.png")

plt.figure()
plt.plot(t, ls, label='3x3 Local Convolution')
plt.plot(t, gs, label='3x3 Global Convolution')
plt.grid()
plt.legend()
plt.xlabel('Samples')
plt.ylabel("Error Rate")
plt.xlim([0, 2500])
plt.ylim([0, 1.0])
plt.savefig("comparison_begin.png")

plt.figure()
plt.plot(t, ls, label='3x3 Local Convolution')
plt.plot(t, gs, label='3x3 Global Convolution')
plt.grid()
plt.legend()
plt.xlabel('Samples')
plt.ylabel("Error Rate")
plt.xlim([50000, 52000])
plt.ylim([0, 1])
plt.savefig("comparison_relabel.png")

plt.figure()
plt.plot(t, ls, label='3x3 Local Convolution')
plt.plot(t, gs, label='3x3 Global Convolution')
plt.grid()
plt.legend()
plt.xlabel('Samples')
plt.ylabel("Error Rate")
plt.xlim([60000, 70000])
plt.ylim([0, 0.4])
plt.savefig("comparison_testSet.png")
