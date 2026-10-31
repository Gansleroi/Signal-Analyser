import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

class PSD:
    def __init__(self, fs: float, signal: np.ndarray) -> None:
        self.fs = fs
        self.signal = signal
    
    def welch(self):
        f, Pxx = signal.welch(self.signal, fs=self.fs, nperseg=1024)
        
        plt.semilogy(f, Pxx)
        plt.xlabel("Частота, Гц")
        plt.ylabel("СПМ")
        plt.grid(True)
        plt.show()