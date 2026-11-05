import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from PyEMD import EMD


class EMDAnalyzer:
    def __init__(self, signal_obj: object) -> None:
        self.t = np.asarray(signal_obj.t)
        self.signal = np.asarray(signal_obj.data)
        self.imfs = None

    def decompose(self, max_imf: int = 4) -> np.ndarray:
        print(f"Выполняется ЭМД (поиск до {max_imf} мод)...")

        emd_analyzer = EMD()
        self.imfs = emd_analyzer.emd(
            self.signal,
            self.t,
            max_imf=max_imf
        )

        print(f"Найдено компонент: {len(self.imfs)}.")
        return self.imfs

    def plot_imfs(self, description: str = "Сложный сигнал") -> None:
        if self.imfs is None:
            raise ValueError("Сначала вызовите метод decompose()!")

        num_imfs = len(self.imfs)

        fig, axs = plt.subplots(
            nrows=num_imfs + 1,
            ncols=1,
            figsize=(12, 3 * (num_imfs + 1)),
            sharex=True
        )

        axs[0].plot(self.t, self.signal, color="tab:blue", linewidth=1)
        axs[0].set_title(
            f"Исходный сигнал: {description}",
            fontsize=14,
            fontweight="bold"
        )
        axs[0].set_ylabel("Амплитуда, В")
        axs[0].grid(True)

        for i in range(num_imfs):
            axs[i + 1].plot(
                self.t, self.imfs[i],
                color="tab:orange",
                linewidth=1.2
            )
            axs[i + 1].text(
                0.01, 0.85, f"IMF {i + 1}",
                transform=axs[i + 1].transAxes,
                fontsize=12,
                fontweight="bold",
                color="darkorange"
            )
            axs[i + 1].set_ylabel("Амплитуда, В")
            axs[i + 1].grid(True)

        axs[-1].set_xlabel("Время, с")

        view_time = self.t[-1] * 0.05
        for ax in axs:
            ax.set_xlim(0, view_time)

        plt.tight_layout()

        folder_name = "plots"
        os.makedirs(folder_name, exist_ok=True)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(folder_name, f"emd_{now}.png")
        plt.savefig(filename, dpi=150)
        print(f"График ЭМД сохранен в файл: {filename}")

        plt.show()

    def plot_imf_spectra(self, max_frequency: float = None) -> None:
        if self.imfs is None:
            raise ValueError("Сначала вызовите метод decompose()!")

        if len(self.t) < 2:
            raise ValueError("Для расчёта спектра нужно минимум две временные точки.")

        dt_values = np.diff(self.t)
        dt = float(np.mean(dt_values))

        if not np.allclose(dt_values, dt, rtol=1e-4, atol=1e-12):
            raise ValueError("Временная сетка должна быть равномерной для FFT.")

        fs = 1.0 / dt
        n = len(self.t)
        window = np.hanning(n) if n > 2 else np.ones(n)

        n_fft = 2 ** int(np.ceil(np.log2(max(2, 4 * n))))
        frequencies = np.fft.rfftfreq(n_fft, d=dt)

        num_imfs = len(self.imfs)
        fig, axs = plt.subplots(
            nrows=num_imfs,
            ncols=1,
            figsize=(12, max(3, 2.8 * num_imfs)),
            squeeze=False
        )
        axs = axs.ravel()

        print("\nДоминирующие частоты IMF:")
        print(f"Частота дискретизации: {fs:.3f} Гц")
        print(f"Номинальное разрешение FFT: {fs / n:.3f} Гц")

        folder_name = "plots"
        os.makedirs(folder_name, exist_ok=True)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for i, imf in enumerate(self.imfs):
            x = np.asarray(imf, dtype=float)
            x = x - np.mean(x)

            spectrum = np.abs(np.fft.rfft(x * window, n=n_fft))
            if len(spectrum) > 0:
                spectrum[0] = 0.0

            peak_index = int(np.argmax(spectrum))
            dominant_frequency = float(frequencies[peak_index])
            peak_amplitude = float(spectrum[peak_index])

            print(
                f"IMF {i + 1}: {dominant_frequency:.2f} Гц "
                f"(амплитуда пика спектра: {peak_amplitude:.4g})"
            )

            axs[i].plot(frequencies, spectrum, linewidth=1.1)
            axs[i].axvline(
                dominant_frequency,
                linestyle="--",
                linewidth=1,
                label=f"Пик: {dominant_frequency:.2f} Гц"
            )
            axs[i].set_title(f"Спектр IMF {i + 1}")
            axs[i].set_ylabel("Амплитуда")
            axs[i].grid(True)
            axs[i].legend()

            if max_frequency is not None:
                axs[i].set_xlim(0, min(max_frequency, fs / 2))
            else:
                axs[i].set_xlim(0, fs / 2)

        axs[-1].set_xlabel("Частота, Гц")
        plt.tight_layout()

        filename = os.path.join(folder_name, f"emd_spectra_{now}.png")
        plt.savefig(filename, dpi=150)
        print(f"Графики спектров IMF сохранены в файл: {filename}")

        plt.show()
