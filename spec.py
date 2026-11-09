import numpy as np

class Spectr:
    def __init__(self, signal_obj, fs: float, use_hann: bool = False) -> None:
        self.t = signal_obj.t
        self.fs = fs
        self.signal = signal_obj.data
        self.n = len(self.signal) // 2
        self.use_hann = use_hann

        if self.use_hann:
            self.window = np.hanning(len(self.signal))
            self.norm_factor = np.mean(self.window[:, None])
        else:
            self.norm_factor = 1.0

    def spec(self) -> np.ndarray:
        if self.signal is None:
            raise ValueError("Сигнал пуст!")

        if self.use_hann:
            windowed_signal = self.signal * self.window
            self.data = np.fft.fft(windowed_signal)
        else:
            self.data = np.fft.fft(self.signal)

        return self.data

    def freqs(self) -> np.ndarray:
        full_freqs = np.fft.fftfreq(len(self.signal), d=1/self.fs)
        self._freqs = full_freqs[:self.n]
        return self._freqs

    def amplspec(self) -> np.ndarray:
        self.spec()

        self.amplitude = np.abs(self.data[:self.n]) / (len(self.signal) * self.norm_factor) * 2
        return self.amplitude

    def freqs_shifted(self) -> np.ndarray:
        full_freqs = np.fft.fftfreq(len(self.signal), d=1/self.fs)
        self._freqs_shifted = np.fft.fftshift(full_freqs)
        return self._freqs_shifted

    def amplspec_shifted(self) -> np.ndarray:
        self.spec()

        full_amplitude = (
            np.abs(self.data) /
            (len(self.signal) * self.norm_factor)
        )

        self.amplitude = np.fft.fftshift(full_amplitude)
        return self.amplitude

    def phasespec_shifted(self) -> np.ndarray:
        self.spec()

        amplitude = np.abs(self.data) / (len(self.signal) * self.norm_factor)
        full_phase = np.degrees(np.angle(self.data))

        threshold = np.max(amplitude) * 1e-6
        full_phase[amplitude < threshold] = np.nan

        self.phase_spectrum = np.fft.fftshift(full_phase)
        return self.phase_spectrum

    def real_spec(self) -> np.ndarray:
        if self.signal is None:
            raise ValueError("Сигнал пуст!")

        sig = self.signal
        if self.use_hann:
            w = self.window if sig.ndim == 1 else self.window[:, None]
            sig = sig * w

        self.data = np.fft.rfft(sig, axis=0)
        return self.data

    def real_amplspec(self) -> np.ndarray:
        self.real_spec()
        amp = np.abs(self.data[:self.n]) / (len(self.signal) * self.norm_factor)
        amp[1:] *= 2
        self.amplitude = amp
        return amp

    def phase_real_spec(self) -> np.ndarray:
        self.real_spec()
        spec = self.data[:self.n]
        amp = np.abs(spec)
        phase = np.degrees(np.angle(spec))

        threshold = amp.max(axis=0, keepdims=True) * 1e-7
        phase[amp < threshold] = np.nan

        self.phase_spectrum = phase
        return phase

    def real_freqs(self) -> np.ndarray:
        self._freqs = np.fft.rfftfreq(len(self.signal), d=1/self.fs)[:self.n]
        return self._freqs
