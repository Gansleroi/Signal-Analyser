import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from autoscale import (choose_freq_limit, choose_time_window,
                       find_spectral_lines, follow_x, log_ylim)


class Plot:
    STEM_LIMIT = 1500
    PHASE_REL_THRESHOLD = 1e-2
    ANNOTATION_REL_THRESHOLD = 0.5
    MAX_ANNOTATIONS = 10

    def __init__(self, fs: float, signal: np.ndarray, amplsp: np.ndarray,
                 phasesp: np.ndarray, t: np.ndarray, freqs: np.ndarray,
                 description: str = "Сложный сигнал",
                 time_window: tuple[float, float] | None = None,
                 freq_limit: float | None = None) -> None:

        self.signal = signal
        self.amplsp = amplsp
        self.phasesp = phasesp
        self.t = t
        self.fs = fs
        self.freqs = freqs
        self.description = description
        self.time_window = time_window
        self.freq_limit = freq_limit

    def plots(self, show: bool = True, save: bool = True):
        fig, axs = plt.subplot_mosaic(
            [["signal", "signal"], ["amp", "phase"]],
            figsize=(14, 10), layout="constrained",
        )

        axs["phase"].sharex(axs["amp"])

        self._plot_signal(axs["signal"])
        self._plot_amplitude(axs["amp"])
        self._plot_phase(axs["phase"], axs["amp"])

        if save:
            self._save(fig)
        if show:
            plt.show()
            plt.close(fig)
        return fig

    def _plot_signal(self, ax) -> None:
        ax.plot(self.t, self.signal, color="tab:blue", linewidth=1.0)
        ax.set_title(f"Сигнал: {self.description}", fontsize=14)
        ax.set_xlabel("Время, с")
        ax.set_ylabel("Амплитуда, В")
        ax.grid(True)

        window = self.time_window or choose_time_window(self.t, self.signal)

        follow_x(ax, margin=0.08)
        ax.set_xlim(*window)

    def _plot_amplitude(self, ax) -> None:
        freqs, amps = np.asarray(self.freqs), np.asarray(self.amplsp)
        limit = self.freq_limit or choose_freq_limit(freqs, amps, self.fs)

        ax.set_yscale("log")
        visible = np.abs(freqs) <= limit
        peak = np.nanmax(amps) if amps.size else 0.0

        if visible.sum() <= self.STEM_LIMIT:

            base = peak * 1e-12 if peak > 0 else 1e-12
            stem = ax.stem(freqs[visible], amps[visible], basefmt=" ",
                           linefmt="tab:orange", markerfmt="o", bottom=base)
            stem.markerline.set_markersize(3)
        else:
            ax.plot(freqs[visible], amps[visible], color="tab:orange",
                    linewidth=0.8)

        ax.set_title("Амплитудный спектр", fontsize=12)
        ax.set_xlabel("Частота, Гц")
        ax.set_ylabel("Амплитуда, В")
        ax.grid(True, which="both", alpha=0.4)

        def y_limits(a):
            lo, hi = sorted(a.get_xlim())
            m = (freqs >= lo) & (freqs <= hi)
            return log_ylim(amps[m])

        follow_x(ax, compute=y_limits)
        ax.set_xlim(-limit, limit)

    def _plot_phase(self, ax, ax_amp) -> None:
        freqs = np.asarray(self.freqs)
        amps = np.asarray(self.amplsp)
        phase = np.asarray(self.phasesp)

        limit = max(abs(v) for v in ax_amp.get_xlim())
        peak = np.nanmax(amps) if amps.size else 0.0
        significant = (
            (np.abs(freqs) <= limit) & np.isfinite(phase)
            & (amps > self.PHASE_REL_THRESHOLD * peak)
        )

        lines = find_spectral_lines(amps, rel_threshold=self.PHASE_REL_THRESHOLD,
                                    max_lines=200)
        if lines is not None:

            lines = lines[significant[lines]]
            stem = ax.stem(freqs[lines], phase[lines], basefmt=" ",
                           linefmt="tab:green", markerfmt="s")
            stem.markerline.set_markersize(3 if len(lines) > 40 else 6)
        else:
            ax.plot(freqs[significant], phase[significant], ".",
                    color="tab:green", markersize=2)

        ax.set_title("Фазовый спектр", fontsize=12)
        ax.set_xlabel("Частота, Гц")
        ax.set_ylabel("Фаза, градусы")
        ax.set_ylim(-190, 190)
        ax.set_yticks(np.arange(-180, 181, 90))
        ax.grid(True)

        #if peak > 0:
            #self._annotate_phase_peaks(ax, freqs, amps, phase, limit, peak)

    def _annotate_phase_peaks(self, ax, freqs, amps, phase, limit, peak):
        idx, _ = find_peaks(amps, height=self.ANNOTATION_REL_THRESHOLD * peak,
                            distance=3)
        idx = idx[(np.abs(freqs[idx]) <= limit) & np.isfinite(phase[idx])]
        order = idx[np.argsort(amps[idx])[::-1]]
        min_gap = 0.04 * 2 * limit
        placed = []

        for i in order:
            f = freqs[i]
            if len(placed) >= self.MAX_ANNOTATIONS:
                break
            if any(abs(f - p) < min_gap for p in placed):
                continue
            placed.append(f)

            value = phase[i] + 90.0 * np.sign(f)
            value = float(np.round((value + 180.0) % 360.0 - 180.0)) + 0.0
            ax.annotate(
                f"{value:.0f}°",
                xy=(f, phase[i]),
                textcoords="offset points",
                xytext=(5 if f > 0 else -5, 10 if value >= 0 else -20),
                ha="left" if f > 0 else "right",
                fontweight="bold", color="darkgreen", fontsize=9, zorder=10,
            )

    @staticmethod
    def _save(fig) -> None:
        folder_name = "plots"
        os.makedirs(folder_name, exist_ok=True)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(folder_name, f"plot_{now}.png")
        fig.savefig(filename, dpi=150)
        print(f"График сохранен в файл: {filename}")
