from .base import BaseSignal
import numpy as np
from scipy.signal import square

class SineSignal(BaseSignal):

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t
        self.data = (self.ampl * np.sin(
            2 * np.pi * self.freq * t + self.phase * np.pi / 180))
        return self.data

class SquareSignal(BaseSignal):

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t
        self.data = self.ampl * square(
            2 * np.pi * self.freq * t +
            self.phase * np.pi / 180
        )
        return self.data

class TriangleSignal(BaseSignal):

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t
        x = 2 * np.pi * self.freq * t + (self.phase * np.pi / 180)
        self.data = self.ampl * (2 / np.pi) * np.arcsin(np.sin(x))
        return self.data

class SawtoothSignal(BaseSignal):

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t
        phase_shift = self.phase / 360.0
        x = (
            (self.freq * t + phase_shift) % 1.0)
        self.data = self.ampl * (2.0 * x - 1.0)
        return self.data
