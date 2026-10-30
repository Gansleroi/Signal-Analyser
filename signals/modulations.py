from .basic import BaseSignal
import numpy as np

class AMSignal(BaseSignal):
    def __init__(self, freq: float, ampl: float, freq_mod: float, mod_index: float = 0.5, phase: float = 0.0) -> None:

        super().__init__(freq=freq, ampl=ampl, phase=phase)
        self.freq_mod = freq_mod
        self.mod_index = mod_index

    def generate(self, t: np.ndarray) -> np.ndarray:

        self.t = t
        envelope = 1 + self.mod_index * np.sin(2 * np.pi * self.freq_mod * t)
        carrier = np.sin(2 * np.pi * self.freq * t + self.phase * np.pi / 180)
        self.data = self.ampl * envelope * carrier
        return self.data

    def __str__(self):
        return f"AM {self.freq / 1000: .1f}кГц (mod {self.freq_mod / 1000: .1f}кГц)"

class ChirpSignal(BaseSignal):
    def __init__(self, freq_start: float, freq_end: float, ampl: float, phase: float = 0.0) -> None:
        super().__init__(freq=freq_start, ampl=ampl, phase=phase)
        self.freq_end = freq_end

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t
        T = t[-1] - t[0]
        k = (self.freq_end - self.freq) / T

        inst_phase = 2 * np.pi * (self.freq * t + (k / 2) * t**2)
        self.data = self.ampl * np.sin(inst_phase + self.phase * np.pi / 180)
        return self.data
    
    def __str__(self):
        return f"Chirp {self.freq/1000:.1f}-{self.freq_end/1000:.1f}кГц"

class PMSignal(BaseSignal):
    def __init__(self, freq: float, ampl: float, freq_mod: float, mod_index: float = 1.0, phase: float = 0.0) -> None:
        super().__init__(freq=freq, ampl=ampl, phase=phase)
        self.freq_mod = freq_mod
        self.mod_index = mod_index

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t
        mod_signal = self.mod_index * np.sin(2 * np.pi * self.freq_mod * t)

        self.data = self.ampl * np.sin(2 * np.pi * self.freq * t + mod_signal + self.phase * np.pi / 180)
        return self.data

    def __str__(self):
        return f"PM {self.freq/1000:.1f}кГц (mod {self.freq_mod/1000:.1f}кГц)"
