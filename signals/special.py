from .base import BaseSignal
import numpy as np

class DampedSineSignal(BaseSignal):
    def __init__(self, freq: float, ampl: float, damping: float, phase: float = 0.0) -> None:
        super().__init__(freq=freq, ampl=ampl, phase=phase)
        self.damping = damping # Коэффициент затухания (например, 1000)

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t
        envelope = np.exp(-self.damping * t) 
        self.data = self.ampl * envelope * np.sin(2 * np.pi * self.freq * t + self.phase * np.pi / 180)
        return self.data

    def __str__(self):
        return f"Damped {self.freq/1000:.1f}кГц"
