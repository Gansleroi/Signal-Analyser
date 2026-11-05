import os
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from autoscale import log_ylim


class PSD:
    def __init__(self, fs: float, sig: np.ndarray) -> None:
        self.fs = fs
        self.sig = sig

    def welch(self, show: bool = True, save: bool = True):
        nperseg = min(1024, len(self.sig))
        f, pxx = signal.welch(self.sig, fs=self.fs, nperseg=nperseg)

        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        ax.semilogy(f, pxx)
        ax.set_title("СПМ сигнала")
        ax.set_xlabel("Частота, Гц")
        ax.set_ylabel("СПМ")
        ax.grid(True, which="both", alpha=0.4)

        ax.set_xlim(f[0], f[-1])
        ax.set_ylim(*log_ylim(pxx, dynamic_range_db=100.0, power=True))

        if save:
            folder_name = "plots"
            os.makedirs(folder_name, exist_ok=True)
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(folder_name, f"psd_{now}.png")
            fig.savefig(filename, dpi=150)
            print(f"График СПМ сохранен в файл: {filename}")

        if show:
            plt.show()
            plt.close(fig)
        return fig
