import soundfile as sf
from signals import ImportSignal
from spec import Spectr
from graph import Plot
import numpy as np
from adc import adc_any_bit, dithering
import matplotlib.pyplot as plt
from PSD import PSD
from exporter import AudioExporter as sw

def imp(file_name: str, start_s: int = 0, dur_s: int = 0, dtype: str = 'float64', mono: bool = False) -> np.ndarray:
    info = sf.info(file_name)
    fs = info.samplerate
    fs = fs

    if dur_s == 0:
        data, _ = sf.read(file_name, dtype=dtype)
    elif dur_s != 0:
        data, _ = sf.read(file_name, start=int(start_s * fs), frames=int(dur_s * fs), dtype=dtype)

    if mono and data.ndim > 1:
        data = data.mean(axis=1)
    t = np.arange(len(data)) / fs

    return data, fs, t



if __name__ == '__main__':

    brams = 'import/brahms-symphony-no-3.flac'
    dorogi = 'import/Dorogy.flac'
    danzon = 'import/Danzon2.flac'

    data, fs, t = imp(file_name=brams, start_s=0, dur_s=0)
    signal = ImportSignal(data=data, fs=fs, t=t)

    spec = Spectr(fs=fs, signal_obj=signal, use_hann=False)

    spec.amplspec_shifted()
    ampl_spec = spec.amplitude
    spec.phasespec_shifted()
    phase_spec = spec.phase_spectrum
    spec.freqs_shifted()
    freqs = spec._freqs_shifted

    pl = Plot(
        fs=fs,
        signal=signal.data,
        amplsp=ampl_spec,
        phasesp=phase_spec,
        freqs=freqs,
        t=t,
        description='Импортированный сигнал',
        time_window=[0.0, 90.0]
    )

    pl.plots(save=False)

    BITS = 16
    dither = False

    if dithering:

        sig_d, v_min, v_max = dithering(signal_obj=signal, bits=BITS)

        digital, restored = adc_any_bit(signal_obj=sig_d, bits=BITS, v_min=v_min, v_max=v_max)

    else:
        digital, restored = adc_any_bit(signal_obj=signal, bits=BITS)

    spec_rest = Spectr(fs=fs, signal_obj=restored, use_hann=False)

    noise = restored.data - signal.data

    spec_rest.amplspec_shifted()
    restored_ampl_spec = spec_rest.amplitude
    spec_rest.phasespec_shifted()
    restored_phase_spec = spec_rest.phase_spectrum
    spec_rest.freqs_shifted()
    restored_freqs = spec_rest._freqs_shifted

    pl_restore = Plot(
        fs=fs,
        signal=restored.data,
        amplsp=restored_ampl_spec,
        phasesp=restored_phase_spec,
        freqs=restored_freqs,
        t=t,
        description='Сигнал после квантования',
        time_window=[0.0, 90.0]
    )

    pl_restore.plots(save=False)

    bits_arr = np.arange(1, 25)
    snr_meas = []

    '''
    for b in bits_arr:
        dig, rest = adc_any_bit(signal, bits=b)
        noise = rest.data - signal.data
        err = signal.data - rest.data
        snr_meas.append(10*np.log10(signal.data.var() / noise.var()))
    
    plt.plot(bits_arr, snr_meas, 'o-', label="Измеренное")
    plt.plot(bits_arr, 6.02 * bits_arr, '--', label="Теория: 6,02·N")
    plt.xlabel("Разрядность АЦП, бит")
    plt.ylabel("ОСШ, дБ")
    plt.legend()
    plt.grid(True)
    plt.show()
    '''

    psd = PSD(fs=fs, sig=signal.data)
    psd.welch(save=False)

    psd_rest = PSD(fs=fs, sig=restored.data)
    psd_rest.welch(save=False)

    psd_noise = PSD(fs=fs, sig=noise)
    psd_noise.welch(save=False)

    data_for_save = restored.data / np.max(np.abs(restored.data))

    sw.save_to_wav(filename=f'Брамс_Квантованный_{BITS}бита.wav', signal=data_for_save, fs=fs)
