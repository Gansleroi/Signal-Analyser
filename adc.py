import numpy as np
import matplotlib.pyplot as plt
from signals import (AMSignal, TriangleSignal, SawtoothSignal,
                     WhiteNoiseSignal)


def adc_3bit(signal: list, v_min: float, v_max: float) -> np.ndarray:
    dig_signal = []
    for i in range(len(signal)):

        signal_val = signal[i]
        if signal_val < v_min:
            signal_val = v_min
        elif signal_val > v_max:
            signal_val = v_max

        num_levels = 2 ** 8

        step = (v_max - v_min) / num_levels

        if signal_val == v_max:
            dig_signal_val = int(num_levels - 1)
            dig_signal.append(dig_signal_val)
        else:
            dig_signal_val = int((signal_val - v_min) / step)
            dig_signal.append(dig_signal_val)

    return dig_signal

fs = 44100
ts = 0.03
N = int(fs * ts)
t = np.arange(N) / fs
freq = 1.0e3
freq_mod = 150.0
ampl=4.0
phase = 90.0
mod_index = 0.8

#signal = SawtoothSignal(freq=freq, ampl=ampl, phase=phase)
signal = AMSignal(ampl=ampl, phase=phase, freq=freq, freq_mod=freq_mod, mod_index=mod_index)
#signal = WhiteNoiseSignal(ampl=ampl)

signal.generate(t)

dig = adc_3bit(signal.data, v_min=-ampl*(1.0+mod_index), v_max=ampl*(1.0+mod_index))

fig, ax = plt.subplots(nrows=2, ncols=1)

ax[0].plot(t, signal.data)
ax[1].plot(t, dig)

plt.show()