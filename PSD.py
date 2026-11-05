import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import os
from datetime import datetime

class PSD:
    def __init__(self, fs: float, sig: np.ndarray) -> None:
        self.fs = fs
        self.sig = sig
    
    def welch(self) -> None:
        f, pxx = signal.welch(self.sig, fs=self.fs, nperseg=1024)
        
        plt.semilogy(f, pxx)
        plt.title("СПМ сигнала")
        plt.xlabel("Частота, Гц")
        plt.ylabel("СПМ")
        plt.grid(True)

        folder_name = "plots"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(folder_name, f"emd_{now}.png")
        plt.savefig(filename, dpi=150)
        print(f"График СПМ сохранен в файл: {filename}")

        plt.show()