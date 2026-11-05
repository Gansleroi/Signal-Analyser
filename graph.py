import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

class Plot:

    def __init__(self, fs: float ,signal: np.ndarray, amplsp: np.ndarray,
                 phasesp: np.ndarray, t: np.ndarray, freqs: np.ndarray, description: str = "Сложный сигнал") -> None:
        self.signal = signal
        self.amplsp = amplsp
        self.phasesp = phasesp
        self.t = t
        self.fs = fs
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
        if max_amp > 0:
            sig_indices = np.where(self.amplsp > (max_amp * 0.05))[0]
            abs_freqs = np.abs(self.freqs[sig_indices])
            nonzero_freqs = abs_freqs[abs_freqs > 0]
            
            if len(nonzero_freqs) > 50:
                view_time = 0.005
            elif len(nonzero_freqs) > 0:
                lowest_freq = np.min(nonzero_freqs)
                diffs = np.diff(np.unique(nonzero_freqs))
                min_diff = np.min(diffs) if len(diffs) > 0 else lowest_freq
                best_freq = min(lowest_freq, min_diff)
                period = 1 / best_freq
                view_time = min(self.t[-1], 5 * period)
            else:
                view_time = min(self.t[-1], 0.01)
                
            axs["signal"].set_xlim(0, view_time)

        limit = min(self.fs / 2, 20000)

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
        threshold = max_peak * 0.5

        if max_peak > 0:

            for i in range(2, len(self.freqs) - 2):

                if self.amplsp[i] <= threshold:
                    continue

                if not (
                        self.amplsp[i] > self.amplsp[i - 1]
                        and self.amplsp[i] >= self.amplsp[i + 1]
                ):
                    continue

                if self.amplsp[i] != np.max(self.amplsp[i - 2:i + 3]):
                    continue

                phase_value = clean_phase[i] + 90

                phase_value = (phase_value + 180) % 360 - 180

                offset = 10 if phase_value >= 0 else -20

                align = 'left' if self.freqs[i] > 0 else 'right'
                shift_x = 5 if self.freqs[i] > 0 else -5

                axs["phase"].annotate(
                    text=f"{phase_value:.0f}°",
                    xy=(self.freqs[i], clean_phase[i]),
                    textcoords="offset points",
                    xytext=(shift_x, offset),
                    ha=align,
                    fontweight='bold',
                    color='darkgreen',
                    fontsize=9,
                    zorder=10
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