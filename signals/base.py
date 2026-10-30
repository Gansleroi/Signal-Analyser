import copy
import numpy as np

class BaseSignal:

    def __init__(self, freq: float, ampl: float, phase: float = 0.0) -> None:
        self.freq: float = freq
        self.ampl: float = ampl
        self.phase: float = phase
        self.t: np.ndarray | None = None
        self.data: np.ndarray | None = None

    def __str__(self):
        name = self.__class__.__name__.replace("Signal", "")
        return f"{name} {self.freq / 1000: .1f}кГц"

    def __add__(self, other):
        new_sig = copy.copy(self)
        new_sig.data = self.data + other.data
        return new_sig

