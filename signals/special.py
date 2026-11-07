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
    
class GaussianSignal(BaseSignal):
    def __init__(self, ampl: float, sigma: float, phase: float = 0.0) -> None:
        super().__init__(freq=0.0, ampl=ampl, phase=phase)
        self.sigma = sigma

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t
        t_center = t[-1] / 2.0 
        envelope = np.exp(-((t - t_center) ** 2) / (2 * self.sigma ** 2))
        self.data = self.ampl * envelope
        return self.data

    def __str__(self):
        return f"Gaussian (σ={self.sigma*1e3:.2f}мс)" 

'''
class WhiteNoiseSignal(BaseSignal):
    def __init__(self, ampl: float) -> None:
        super().__init__(freq=0.0, ampl=ampl)

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t
        self.data = np.random.normal(0, self.ampl, len(t))
        return self.data

    def __str__(self):
        return f"White Noise (σ={self.ampl})"


class BrownianNoiseSignal(BaseSignal):
    def __init__(self, ampl: float) -> None:
        super().__init__(freq=0.0, ampl=ampl)

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t
        N = len(t)
        steps = np.random.normal(0, 1, N)
        walk = np.cumsum(steps)

        walk = walk - np.linspace(walk[0], walk[-1], N)

        max_val = np.max(np.abs(walk))
        if max_val > 0:
            self.data = (walk / max_val) * self.ampl
        else:
            self.data = walk
        return self.data

    def __str__(self):
        return f"Brownian Noise (σ={self.ampl})"


class PinkNoiseSignal(BaseSignal):
    def __init__(self, ampl: float) -> None:
        super().__init__(freq=0.0, ampl=ampl)

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t
        N = len(t)
        freqs = np.fft.fftfreq(N, d=t[1]-t[0])
        freqs[0] = 1 
        
        white = np.random.normal(0, 1, N) + 1j * np.random.normal(0, 1, N)

        pink = white / np.sqrt(np.abs(freqs))
        pink[0] = 0
 
        self.data = np.real(np.fft.ifft(pink))

        max_val = np.max(np.abs(self.data))
        if max_val > 0:
            self.data = (self.data / max_val) * self.ampl
        return self.data

    def __str__(self):
        return f"Pink Noise (1/f, σ={self.ampl})"   
'''