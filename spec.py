import numpy as np

class Spectr:
    def __init__(self, signal_obj, use_hann: bool = False) -> None:
        self.t = signal_obj.t
        self.signal = signal_obj.data
        self.n = len(self.t) // 2
        self.use_hann = use_hann

        if self.use_hann:
            self.window = np.hanning(len(self.signal))
            self.norm_factor = np.mean(self.window)
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
        full_freqs = np.fft.fftfreq(len(self.t), d=self.t[1] - self.t[0])
        self.freqs = full_freqs[:self.n]
        return self.freqs

    def amplspec(self) -> np.ndarray:
        if not hasattr(self, 'data'):
            self.spec()

        self.amplitude = np.abs(self.data[:self.n]) / (len(self.t) * self.norm_factor) * 2
        return self.amplitude

    def freqs_shifted(self) -> np.ndarray:
        full_freqs = np.fft.fftfreq(len(self.t), d=self.t[1] - self.t[0])
        self.freqs = np.fft.fftshift(full_freqs)
        return self.freqs

    def amplspec_shifted(self) -> np.ndarray:
        if not hasattr(self, 'data'):
            self.spec()
        full_amplitude = np.abs(self.data) / (len(self.t) * self.norm_factor) * 2
        self.amplitude = np.fft.fftshift(full_amplitude)
        return self.amplitude

    def phasespec_shifted(self) -> np.ndarray:
        if not hasattr(self, 'data'):
            self.spec()
        full_phase = np.degrees(np.angle(self.data))
        self.phase = np.fft.fftshift(full_phase)
        return self.phase
