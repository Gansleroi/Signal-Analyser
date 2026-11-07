from .basic import BaseSignal
import numpy as np

class AMSignal(BaseSignal):
    def __init__(self, freq: float, ampl: float, freq_mod: float,
                 mod_index: float = 0.5, phase: float = 0.0) -> None:
        super().__init__(freq=freq, ampl=ampl, phase=phase)
        self.freq_mod = freq_mod
        self.mod_index = mod_index

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t

        envelope = (
                1.0 + self.mod_index *
                np.sin(2 * np.pi * self.freq_mod * t)
        )

        carrier = np.sin(
            2 * np.pi * self.freq * t
            + np.deg2rad(self.phase)
        )

        self.data = self.ampl * envelope * carrier

        return self.data

    def __str__(self) -> str:
        if self.freq < 1000:
            carrier_str = f"{self.freq:.0f}Гц"
        else:
            carrier_str = f"{self.freq/1000:.1f}кГц"

        if self.freq_mod < 1000:
            mod_str = f"{self.freq_mod:.0f}Гц"
        else:
            mod_str = f"{self.freq_mod/1000:.1f}кГц"
            
        return f"AM {carrier_str} (mod {mod_str})"

class ChirpSignal(BaseSignal):
    def __init__(self, freq_start: float, freq_end: float,
                 ampl: float, phase: float = 0.0) -> None:
        super().__init__(freq=freq_start, ampl=ampl, phase=phase)
        self.freq_end = freq_end

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t
        T = t[-1] - t[0]
        k = (self.freq_end - self.freq) / T

        inst_phase = 2 * np.pi * (self.freq * t + (k / 2) * t**2)
        self.data = self.ampl * np.sin(inst_phase + self.phase * np.pi / 180)
        return self.data
    
    def __str__(self) -> str:
        return f"Chirp {self.freq/1000:.1f}-{self.freq_end/1000:.1f}кГц"

class PMSignal(BaseSignal):
    def __init__(self, freq: float, ampl: float, freq_mod: float,
                 mod_index: float = 1.0, phase: float = 0.0) -> None:
        super().__init__(freq=freq, ampl=ampl, phase=phase)
        self.freq_mod = freq_mod
        self.mod_index = mod_index

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t
        mod_signal = self.mod_index * np.sin(2 * np.pi * self.freq_mod * t)

        self.data = self.ampl * np.sin(2 * np.pi * self.freq * t + mod_signal + self.phase * np.pi / 180)
        return self.data

    def __str__(self) -> str:
        mod_str = f"{self.freq_mod:.0f}Гц" if self.freq_mod < 1000 else f"{self.freq_mod/1000:.1f}кГц"
        return f"PM {self.freq/1000:.1f}кГц (mod {mod_str})"

class FMSignal(BaseSignal):

    def __init__(self, freq: float, freq_car: float, freq_dev: float, ampl: float
                 ,ampl_car: float, phase: float = 0.0) -> None:
        super().__init__(freq=freq, ampl=ampl, phase=phase)
        self.freq_car = freq_car
        self.freq_dev = freq_dev
        self.ampl_car = ampl_car

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t

        Df = self.freq_dev / self.ampl
        phi = np.deg2rad(self.phase)

        if np.isclose(self.freq, 0):
            integrad = self.ampl * np.sin(phi) * t
        else:
            integrad = (
                    self.ampl / (2 * np.pi * self.freq)
                    * (np.cos(phi) -
                       np.cos(2 * np.pi * self.freq * t + phi))
            )

        self.data = self.ampl_car * np.cos(
            2 * np.pi * self.freq_car * t
            + 2 * np.pi * Df * integrad
        )

        return self.data

    def __str__(self) -> str:
        return f"FM {self.freq/1000:.1f}кГц"

