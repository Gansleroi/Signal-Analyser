import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

class Plot:

    def __init__(self, signal: np.ndarray, amplsp: np.ndarray, phasesp: np.ndarray, t: np.ndarray, freqs: np.ndarray, description: str = "Сложный сигнал") -> None:
        self.signal = signal
        self.amplsp = amplsp
        self.phasesp = phasesp
        self.t = t
        self.freqs = freqs
        self.n = len(t) // 2
        self.description = description

    def plots(self) -> None:

        mosaic = [
            ["signal", "signal"],
            ["amp", "phase"]
        ]

        fig, axs = plt.subplot_mosaic(mosaic, figsize=(14, 10))

        # --- 1. ВЕРХНИЙ ГРАФИК (Сигнал) ---
        axs["signal"].plot(self.t, self.signal, color='tab:blue')
        axs["signal"].set_title(f"Сигнал: {self.description}", fontsize=14)
        axs["signal"].set_xlabel("Время, с")
        axs["signal"].set_ylabel("Амплитуда, В")
        axs["signal"].grid(True)

        axs["amp"].stem(self.freqs, self.amplsp, basefmt=" ", linefmt='tab:orange', markerfmt='o')
        axs["amp"].set_title("Амплитудный спектр", fontsize=12)
        axs["amp"].set_xlabel("Частота, Гц")
        axs["amp"].set_ylabel("Амплитуда, В")
        axs["amp"].grid(True)
        axs["amp"].set_yscale('log')
        max_amp = np.max(self.amplsp)


        significant_indices = np.where(self.amplsp > (max_amp * 0.05))[0]

        if len(significant_indices) > 0:
            max_freq = np.max(np.abs(self.freqs[significant_indices]))
            limit = max_freq * 1.5
        else:
            limit = 10000

        limit = min(limit, 20000)

        axs["amp"].set_xlim(-limit, limit)
        axs["phase"].set_xlim(-limit, limit)

        clean_phase = self.phasesp.copy()
        clean_phase[self.amplsp < 0.1] = 0

        axs["phase"].stem(self.freqs, clean_phase, basefmt=" ", linefmt='tab:green', markerfmt='s')
        axs["phase"].set_title("Фазовый спектр", fontsize=12)
        axs["phase"].set_xlabel("Частота, Гц")
        axs["phase"].set_ylabel("Фаза, градусы")
        axs["phase"].set_ylim(-190, 190)
        axs["phase"].grid(True)
        axs["phase"].set_xlim(-10000, 10000)

        max_peak = np.max(self.amplsp)
        threshold = max_peak * 0.05 

        for i in range(2, len(self.freqs) - 2):
            if self.amplsp[i] > threshold:
                if self.amplsp[i] == np.max(self.amplsp[i - 2 : i + 3]):
                    offset = 10 if clean_phase[i] >= 0 else -20
                    align = 'left' if self.freqs[i] > 0 else 'right'
                    shift_x = 5 if self.freqs[i] > 0 else -5
                    axs["phase"].annotate(
                        f"{clean_phase[i]: .0f}°",
                        (self.freqs[i], clean_phase[i]),
                        textcoords="offset points",
                        xytext=(shift_x, offset),
                        ha=align,
                        fontweight='bold',
                        color='darkgreen'
                    )

        plt.tight_layout()

        folder_name = "plots"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(folder_name, f"plot_{now}.png")
        plt.savefig(filename, dpi=150)
        print(f"График сохранен в файл: {filename}")

        plt.show()