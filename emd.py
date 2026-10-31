import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from PyEMD import EMD

class EMDAnalyzer:
    def __init__(self, signal_obj) -> None:
        self.t = signal_obj.t
        self.signal = signal_obj.data
        self.imfs = None

    def decompose(self, max_imf: int = 4) -> np.ndarray:
        print(f"Выполняется ЭМД (поиск до {max_imf} мод)...")
        emd = EMD()
        self.imfs = emd.emd(self.signal, self.t, max_imf=max_imf)
        print(f"Найдено {len(self.imfs)} IMF.")
        return self.imfs

    def plot_imfs(self, description: str = "Сложный сигнал") -> None:
        if self.imfs is None:
            raise ValueError("Сначала вызовите метод decompose()!")

        num_imfs = len(self.imfs)
        
        fig, axs = plt.subplots(nrows=num_imfs + 1, ncols=1, 
                                figsize=(12, 3 * (num_imfs + 1)), 
                                sharex=True)
        
        axs[0].plot(self.t, self.signal, color='tab:blue', linewidth=1)
        axs[0].set_title(f"Исходный сигнал: {description}", fontsize=14, fontweight='bold')
        axs[0].set_ylabel("Амплитуда, В")
        axs[0].grid(True)

        for i in range(num_imfs):
            axs[i + 1].plot(self.t, self.imfs[i], color='tab:orange', linewidth=1.2)
            axs[i + 1].text(0.01, 0.85, f"IMF {i + 1}", transform=axs[i + 1].transAxes, 
                        fontsize=12, fontweight='bold', color='darkorange')
            axs[i + 1].set_ylabel("Амплитуда, В")
            axs[i + 1].grid(True)

        axs[-1].set_xlabel("Время, с")
        
        view_time = self.t[-1] * 0.05
        for ax in axs:
            ax.set_xlim(0, view_time)

        plt.tight_layout()

        folder_name = "plots"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(folder_name, f"emd_{now}.png")
        plt.savefig(filename, dpi=150)
        print(f"График ЭМД сохранен в файл: {filename}")

        plt.show()