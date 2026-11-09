import numpy as np
import matplotlib.pyplot as plt
from signals import (AMSignal, TriangleSignal, SawtoothSignal,
                     WhiteNoiseSignal, BlueNoiseSignal,
                     PinkNoiseSignal, PerlinNoiseSignal,
                     BrownianNoiseSignal, VioletNoiseSignal,
                     HumNoiseSignal, ShotNoiseSignal)

from spec import Spectr
import copy
from PSD import PSD
from exporter import AudioExporter
from math import comb


def adc_any_bit(signal_obj, bits: int, v_min: float = None, v_max: float = None):
    analog_signal = np.asarray(signal_obj.data, dtype=float)

    if v_min is None:
        v_min = analog_signal.min()
    if v_max is None:
        v_max = analog_signal.max()

    num_levels = 2 ** bits
    step = (v_max - v_min) / num_levels

    clipped = np.clip(analog_signal, v_min, v_max)

    codes = np.minimum(((clipped - v_min) / step).astype(int), num_levels - 1)

    restored_voltage = v_min + (codes + 0.5) * step

    digital_signal = copy.deepcopy(signal_obj)
    restored_signal = copy.deepcopy(signal_obj)
    digital_signal.data = codes
    restored_signal.data = restored_voltage

    return digital_signal, restored_signal

def adc_any_bit_ns(signal_obj, bits: int, v_min: float = None, v_max: float = None, order: int = 1):
    x = np.asarray(signal_obj.data, dtype=float)

    if v_min is None:
        v_min = x.min()
    if v_max is None:
        v_max = x.max()

    num_levels = 2 ** bits
    step = (v_max - v_min) / num_levels

    # коэффициенты обратной связи: для order=3 -> [3, -3, 1]
    coef = [(-1) ** (k + 1) * comb(order, k) for k in range(1, order + 1)]

    codes = np.empty(len(x), dtype=np.int64)
    restored_voltage = np.empty(len(x), dtype=float)
    errs = [0.0] * order          # errs[0] = e[n-1], errs[1] = e[n-2], ...

    for n in range(len(x)):
        v = x[n]
        for k in range(order):
            v += coef[k] * errs[k]       # даёт -3e1 + 3e2 - e3 для order=3

        c = int((min(max(v, v_min), v_max) - v_min) / step)
        c = min(c, num_levels - 1)
        q = v_min + (c + 0.5) * step

        errs = [q - v] + errs[:-1]       # сдвиг истории ошибок

        codes[n] = c
        restored_voltage[n] = q

    digital_signal = copy.deepcopy(signal_obj)
    restored_signal = copy.deepcopy(signal_obj)
    digital_signal.data = codes
    restored_signal.data = restored_voltage
    return digital_signal, restored_signal


def dithering(bits: int, signal_obj: object, v_min: float = None, v_max: float = None, seed: int = 0):
    x = np.asarray(signal_obj.data, dtype=float)

    if v_min is None:
        v_min = x.min()
    if v_max is None:
        v_max = x.max()
    step = (v_max - v_min) / 2 ** bits

    rng = np.random.default_rng(seed)
    # TPDF: сумма двух равномерных шумов, форма совпадает с x
    dither = (rng.uniform(-step / 2, step / 2, x.shape)
              + rng.uniform(-step / 2, step / 2, x.shape))

    sig_d = copy.deepcopy(signal_obj)
    sig_d.data = x + dither
    return sig_d, v_min, v_max

if __name__ == '__main__':

    fs = 44100
    ts = 3
    N = int(fs * ts)
    t = np.arange(N) / fs
    freq = 1.0e3
    freq_mod = 150.0
    ampl= 6.0
    phase = 90.0
    mod_index = 0.8

    #signal = SawtoothSignal(freq=freq, ampl=ampl, phase=phase)
    #signal = AMSignal(ampl=ampl, phase=phase, freq=freq, freq_mod=freq_mod, mod_index=mod_index)
    #signal = WhiteNoiseSignal(ampl=ampl)
    #signal = BlueNoiseSignal(ampl=ampl)
    #signal = PerlinNoiseSignal(ampl=ampl, freq=440)
    #signal = BrownianNoiseSignal(ampl=ampl)
    #signal = PinkNoiseSignal(ampl=ampl)
    #signal = VioletNoiseSignal(ampl=ampl)
    #signal = HumNoiseSignal()
    signal = ShotNoiseSignal(ampl=ampl)

    signal.generate(t)

    digital, restored = adc_any_bit(signal_obj=signal, bits=3)

    spec_analog_sig = Spectr(signal_obj=signal, fs=fs, use_hann=True)
    spec_analog_sig.amplspec()
    ampl_spec_analog_sig = spec_analog_sig.amplitude
    spec_analog_sig.freqs()
    freqs_spec_analog_sig = spec_analog_sig._freqs

    spec_digital_signal = Spectr(signal_obj=digital, fs=fs, use_hann=True)
    spec_digital_signal.amplspec()
    ampl_spec_digital_signal = spec_digital_signal.amplitude
    spec_digital_signal.freqs()
    freqs_spec_digital_signal = spec_digital_signal._freqs

    spec_restored_signal = Spectr(signal_obj=restored, fs=fs, use_hann=True)
    spec_restored_signal.amplspec()
    ampl_spec_restored_signal = spec_restored_signal.amplitude
    spec_restored_signal.freqs()
    freqs_spec_restored_signal = spec_restored_signal._freqs


    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(20, 6))

    ax[0, 0].plot(t, signal.data)
    ax[0, 0].set_xlabel("Время, с")
    ax[0, 0].set_ylabel("Амплитуда, В")
    ax[0, 0].set_title("Исходный сигнал")
    ax[0, 0].set_xlim(0, 1)
    ax[0, 0].grid(True)

    ax[0, 1].plot(freqs_spec_analog_sig, ampl_spec_analog_sig)
    ax[0, 1].set_xlabel("Частота, Гц")
    ax[0, 1].set_ylabel("Амплитуда, В")
    ax[0, 1].set_title("Амплитудный спектр")
    ax[0, 1].set_xlim(0, 5000)
    ax[0, 1].set_yscale("log")
    ax[0, 1].grid(True)

    ax[1, 0].plot(t, digital.data)
    ax[1, 0].set_xlabel("Время, с")
    ax[1, 0].set_ylabel("Амплитуда ед. АЦП")
    ax[1, 0].set_title("Оцифрованный сигнал")
    ax[1, 0].set_xlim(0, 1)
    ax[1, 0].grid(True)

    ax[1, 1].plot(freqs_spec_digital_signal, ampl_spec_digital_signal)
    ax[1, 1].set_xlabel("Частота, Гц")
    ax[1, 1].set_ylabel("Амплитуда, В")
    ax[1, 1].set_title("Амплитудный спектр")
    ax[1, 1].set_xlim(0, 5000)
    ax[1, 1].set_yscale("log")
    ax[1, 1].grid(True)

    ax[2, 0].plot(t, restored.data)
    ax[2, 0].set_xlabel("Время, с")
    ax[2, 0].set_ylabel("Амплитуда, В")
    ax[2, 0].set_title("Восстановленный сигнал")
    ax[2, 0].set_xlim(0, 1)
    ax[2, 0].grid(True)

    ax[2, 1].plot(freqs_spec_restored_signal, ampl_spec_restored_signal)
    ax[2, 1].set_xlabel("Частота, Гц")
    ax[2, 1].set_ylabel("Амплитуда, В")
    ax[2, 1].set_title("Амплитудный спектр")
    ax[2, 1].set_xlim(0, 5000)
    ax[2, 1].set_yscale("log")
    ax[2, 1].grid(True)

    plt.tight_layout()
    plt.show()

    err = signal.data - restored.data
    snr = 10 * np.log10(np.var(signal.data) / np.var(err))
    print(f"Шаг квантования: {(signal.data.max() - signal.data.min()) / 8:.3f} В")
    print(f"Макс. ошибка: {np.abs(err).max():.3f} В")
    print(f"ОСШ квантования: {snr:.1f} дБ")

    bits_arr = np.arange(1, 25)
    snr_meas = []
    for b in bits_arr:
        dig, rest = adc_any_bit(signal, bits=b)
        err = signal.data - rest.data
        snr_meas.append(10 * np.log10(np.var(signal.data) / np.var(err)))

    plt.plot(bits_arr, snr_meas, 'o-', label="Измеренное")
    plt.plot(bits_arr, 6.02 * bits_arr, '--', label="Теория: 6,02·N")
    plt.xlabel("Разрядность АЦП, бит")
    plt.ylabel("ОСШ, дБ")
    plt.legend()
    plt.grid(True)
    plt.show()

    psd = PSD(fs=fs, sig=signal.data)
    psd.welch()

    AudioExporter.save_to_wav(filename='yfdrf.wav', fs=fs, signal=signal.data)
