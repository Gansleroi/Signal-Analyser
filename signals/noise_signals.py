from .base import BaseSignal
import numpy as np


# ---------------------------------------------------------------------------
# Цветные шумы (спектр мощности ~ 1/f^beta)
# Единая нормировка: СКО (σ) сигнала = ampl
# ---------------------------------------------------------------------------
class _ColoredNoiseSignal(BaseSignal):
    """Базовый класс: шум со спектром мощности 1/f^beta."""
    beta = 0.0
    label = "Noise"

    def __init__(self, ampl: float, seed: int | None = None) -> None:
        super().__init__(freq=0.0, ampl=ampl)
        self.rng = np.random.default_rng(seed)

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t
        N = len(t)
        spectrum = np.fft.rfft(self.rng.normal(0, 1, N))

        freqs = np.fft.rfftfreq(N)
        freqs[0] = freqs[1] if N > 1 else 1.0   # чтобы не делить на 0
        spectrum *= freqs ** (-self.beta / 2.0)
        spectrum[0] = 0                          # убираем постоянную составляющую

        x = np.fft.irfft(spectrum, N)
        std = np.std(x)
        self.data = self.ampl * x / std if std > 0 else x
        return self.data

    def __str__(self):
        return f"{self.label} (σ={self.ampl})"


class WhiteNoiseSignal(_ColoredNoiseSignal):
    beta = 0.0
    label = "White Noise"

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t
        self.data = self.rng.normal(0, self.ampl, len(t))
        return self.data


class PinkNoiseSignal(_ColoredNoiseSignal):
    beta = 1.0
    label = "Pink Noise (1/f"

    def __str__(self):
        return f"Pink Noise (1/f, σ={self.ampl})"


class BrownianNoiseSignal(_ColoredNoiseSignal):
    beta = 2.0
    label = "Brownian Noise (1/f²"

    def __str__(self):
        return f"Brownian Noise (1/f², σ={self.ampl})"


class BlueNoiseSignal(_ColoredNoiseSignal):
    beta = -1.0

    def __str__(self):
        return f"Blue Noise (f, σ={self.ampl})"


class VioletNoiseSignal(_ColoredNoiseSignal):
    beta = -2.0

    def __str__(self):
        return f"Violet Noise (f², σ={self.ampl})"


class CustomColorNoiseSignal(_ColoredNoiseSignal):
    """Шум с произвольным показателем beta (спектр 1/f^beta)."""

    def __init__(self, ampl: float, beta: float, seed: int | None = None) -> None:
        super().__init__(ampl, seed)
        self.beta = beta

    def __str__(self):
        return f"1/f^{self.beta:g} Noise (σ={self.ampl})"


# ---------------------------------------------------------------------------
# Статистические шумы
# ---------------------------------------------------------------------------
class UniformNoiseSignal(BaseSignal):
    """Равномерный шум. ampl — СКО (полуширина = ampl·√3)."""

    def __init__(self, ampl: float, seed: int | None = None) -> None:
        super().__init__(freq=0.0, ampl=ampl)
        self.rng = np.random.default_rng(seed)

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t
        half = self.ampl * np.sqrt(3)
        self.data = self.rng.uniform(-half, half, len(t))
        return self.data

    def __str__(self):
        return f"Uniform Noise (σ={self.ampl})"


class ImpulseNoiseSignal(BaseSignal):
    """Импульсный шум («соль и перец»): выбросы ±ampl с вероятностью density."""

    def __init__(self, ampl: float, density: float = 0.01,
                 salt_ratio: float = 0.5, seed: int | None = None) -> None:
        super().__init__(freq=0.0, ampl=ampl)
        self.density = density
        self.salt_ratio = salt_ratio
        self.rng = np.random.default_rng(seed)

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t
        r = self.rng.random(len(t))
        data = np.zeros(len(t))
        data[r < self.density * self.salt_ratio] = self.ampl
        data[(r >= self.density * self.salt_ratio) & (r < self.density)] = -self.ampl
        self.data = data
        return self.data

    def __str__(self):
        return f"Impulse Noise (p={self.density}, ±{self.ampl})"


class ShotNoiseSignal(BaseSignal):
    """Дробовой шум (Пуассон). rate — среднее число событий на отсчёт;
    чем меньше rate, тем сильнее относительный шум. СКО = ampl."""

    def __init__(self, ampl: float, rate: float = 10.0, seed: int | None = None) -> None:
        super().__init__(freq=0.0, ampl=ampl)
        self.rate = rate
        self.rng = np.random.default_rng(seed)

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t
        counts = self.rng.poisson(self.rate, len(t))
        self.data = self.ampl * (counts - self.rate) / np.sqrt(self.rate)
        return self.data

    def __str__(self):
        return f"Shot Noise (λ={self.rate}, σ={self.ampl})"


# ---------------------------------------------------------------------------
# Наводки
# ---------------------------------------------------------------------------
class HumNoiseSignal(BaseSignal):
    """Сетевая наводка: основная частота + гармоники (50 Гц в Европе)."""

    def __init__(self, freq: float = 50.0, ampl: float = 1.0,
                 harmonics: int = 3, phase: float = 0.0) -> None:
        super().__init__(freq=freq, ampl=ampl, phase=phase)
        self.harmonics = harmonics

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t
        data = np.zeros(len(t))
        for k in range(1, self.harmonics + 1):
            data += (1.0 / k) * np.sin(2 * np.pi * k * self.freq * t
                                       + self.phase * np.pi / 180)
        self.data = self.ampl * data
        return self.data

    def __str__(self):
        return f"Hum {self.freq:.0f}Гц (+{self.harmonics - 1} гарм.)"


# ---------------------------------------------------------------------------
# Процедурный (градиентный) шум Перлина, 1D
# ---------------------------------------------------------------------------
class PerlinNoiseSignal(BaseSignal):
    """1D шум Перлина с октавами (fBm).
    freq — характерная частота изменений (Гц) базовой октавы.
    Нормировка по пику: max|x| = ampl."""

    def __init__(self, freq: float, ampl: float, octaves: int = 1,
                 lacunarity: float = 2.0, persistence: float = 0.5,
                 seed: int | None = None) -> None:
        super().__init__(freq=freq, ampl=ampl)
        self.octaves = octaves
        self.lacunarity = lacunarity
        self.persistence = persistence
        self.seed = seed

    @staticmethod
    def _fade(x):
        return x ** 3 * (x * (x * 6 - 15) + 10)

    def _perlin(self, t: np.ndarray, freq: float, rng) -> np.ndarray:
        pos = (t - t[0]) * freq
        i0 = np.floor(pos).astype(int)
        frac = pos - i0
        grads = rng.uniform(-1, 1, i0.max() + 2)
        g0, g1 = grads[i0], grads[i0 + 1]
        n0 = g0 * frac
        n1 = g1 * (frac - 1)
        u = self._fade(frac)
        return n0 + u * (n1 - n0)

    def generate(self, t: np.ndarray) -> np.ndarray:
        self.t = t
        rng = np.random.default_rng(self.seed)
        data = np.zeros(len(t))
        amp, f = 1.0, self.freq
        for _ in range(self.octaves):
            data += amp * self._perlin(t, f, rng)
            amp *= self.persistence
            f *= self.lacunarity

        peak = np.max(np.abs(data))
        self.data = self.ampl * data / peak if peak > 0 else data
        return self.data

    def __str__(self):
        return f"Perlin {self.freq:.0f}Гц ({self.octaves} окт.)"
