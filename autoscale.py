from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
from scipy.signal import find_peaks

Limits = tuple[float, float]

_NICE_STEPS = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)

def nice_ceil(x: float) -> float:

    if not np.isfinite(x) or x <= 0:
        return 1.0
    exp = np.floor(np.log10(x))
    base = x / 10.0 ** exp
    for step in _NICE_STEPS:
        if base <= step * (1 + 1e-9):
            return step * 10.0 ** exp
    return 10.0 ** (exp + 1)


def padded_limits(lo: float, hi: float, margin: float = 0.05,
                  symmetric: bool = False) -> Limits:

    if not (np.isfinite(lo) and np.isfinite(hi)):
        return -1.0, 1.0
    if lo > hi:
        lo, hi = hi, lo

    scale = max(abs(lo), abs(hi))

    if symmetric:
        m = (scale if scale > 0 else 1.0) * (1 + margin)
        return -m, m

    span = hi - lo
    if span <= 1e-12 * scale:
        half = 0.1 * scale if scale > 0 else 1.0
        return lo - half, hi + half

    pad = margin * span
    return lo - pad, hi + pad

def _data_lines(ax):
    for line in ax.get_lines():
        if line.get_visible() and line.get_transform() is ax.transData:
            yield line


def visible_y_range(ax) -> Limits | None:

    x_lo, x_hi = sorted(ax.get_xlim())
    lo, hi = np.inf, -np.inf

    for line in _data_lines(ax):
        x = np.asarray(line.get_xdata(orig=False), dtype=float)
        y = np.asarray(line.get_ydata(orig=False), dtype=float)
        if x.size == 0 or x.size != y.size:
            continue

        mask = (x >= x_lo) & (x <= x_hi) & np.isfinite(y)
        if mask.any():
            lo = min(lo, float(y[mask].min()))
            hi = max(hi, float(y[mask].max()))

        if x.size > 1 and x[0] < x[-1]:
            for edge in (x_lo, x_hi):
                if x[0] < edge < x[-1]:
                    y_edge = float(np.interp(edge, x, y))
                    if np.isfinite(y_edge):
                        lo, hi = min(lo, y_edge), max(hi, y_edge)

    return (lo, hi) if np.isfinite(lo) and np.isfinite(hi) else None


def linear_y_limits(ax, margin: float = 0.05, symmetric: bool = False,
                    clip: tuple[float | None, float | None] = (None, None)
                    ) -> Limits | None:

    rng = visible_y_range(ax)
    if rng is None:
        return None
    lo, hi = padded_limits(*rng, margin=margin, symmetric=symmetric)
    if clip[0] is not None:
        lo = max(lo, clip[0])
    if clip[1] is not None:
        hi = min(hi, clip[1])
    return (lo, hi) if hi > lo else None


def log_ylim(values: np.ndarray, dynamic_range_db: float = 100.0,
             power: bool = False, min_decades: float = 2.0,
             headroom: float = 2.5) -> Limits:

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return 1e-12, 1.0

    peak = float(v.max())
    k = 10.0 if power else 20.0
    floor = peak * 10.0 ** (-dynamic_range_db / k)

    shown_min = float(v[v >= floor].min())
    low = min(shown_min, peak * 10.0 ** (-min_decades))
    bottom = 10.0 ** np.floor(np.log10(low))
    return bottom, peak * headroom


def follow_x(axes, compute: Callable[[object], Limits | None] | None = None,
             **linear_kwargs) -> list[int]:

    axes = [axes] if hasattr(axes, "get_xlim") else list(axes)
    if compute is None:
        def compute(ax):  # noqa: E306
            return linear_y_limits(ax, **linear_kwargs)

    def _update(_changed_ax=None):
        for ax in axes:
            limits = compute(ax)
            if limits is not None:
                ax.set_ylim(*limits)

    cids = [ax.callbacks.connect("xlim_changed", _update) for ax in axes]
    _update()
    return cids



def choose_freq_limit(freqs: np.ndarray, amps: np.ndarray, fs: float,
                      rel_threshold: float = 1e-3,
                      margin: float = 1.25) -> float:

    nyq = fs / 2.0
    f = np.abs(np.asarray(freqs, dtype=float))
    a = np.asarray(amps, dtype=float)
    ok = np.isfinite(a)
    if not ok.any() or a[ok].max() <= 0:
        return nyq

    df = fs / max(len(f), 1)
    significant = ok & (a > rel_threshold * a[ok].max())
    f_hi = float(f[significant].max())

    limit = nice_ceil(f_hi * margin) if f_hi > 0 else 20 * df
    return float(min(nyq, max(limit, 20 * df)))


def find_spectral_lines(amps: np.ndarray, rel_threshold: float = 0.05,
                        max_lines: int = 20, bins_per_line: int = 8):

    a = np.asarray(amps, dtype=float)
    if a.size == 0 or not np.isfinite(a).all() or a.max() <= 0:
        return None
    threshold = rel_threshold * a.max()
    peaks, _ = find_peaks(a, height=threshold, distance=3)
    if not 1 <= len(peaks) <= max_lines:
        return None
    if np.count_nonzero(a > threshold) > bins_per_line * len(peaks):
        return None
    return peaks


def _active_interval(x: np.ndarray, rel_threshold: float = 0.01):

    dev = np.abs(x - np.median(x))
    peak = dev.max()
    if not np.isfinite(peak) or peak == 0:
        return None
    idx = np.flatnonzero(dev >= rel_threshold * peak)
    return int(idx[0]), int(idx[-1])


def choose_time_window(t: np.ndarray, x: np.ndarray, periods: float = 5.0,
                       max_lines: int = 20, line_threshold: float = 0.05,
                       broadband_samples: int = 1000,
                       min_samples: int = 20) -> Limits:

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(t)
    t_lo, t_hi = float(t[0]), float(t[-1])
    if n < 8 or not np.isfinite(x).all():
        return t_lo, t_hi

    dt = (t_hi - t_lo) / (n - 1)
    min_span = min_samples * dt

    active = _active_interval(x)
    if active is not None:
        i0, i1 = active
        if (i1 - i0 + 1) < 0.5 * n:
            span = max((i1 - i0) * dt, min_span)
            pad = 0.1 * span
            return max(t_lo, t[i0] - pad), min(t_hi, t[i0] + span + pad)

    xc = x - x.mean()
    spectrum = np.abs(np.fft.rfft(xc * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=dt)
    peaks = find_spectral_lines(spectrum, rel_threshold=line_threshold,
                                max_lines=max_lines)
    if peaks is not None:
        f_lines = np.sort(freqs[peaks][freqs[peaks] > 0])
        if f_lines.size:
            base = f_lines[0]
            if f_lines.size > 1:
                base = min(base, float(np.diff(f_lines).min()))
            view = periods / base
            return t_lo, t_lo + float(np.clip(view, min_span, t_hi - t_lo))

    view = broadband_samples * dt
    return t_lo, t_lo + float(np.clip(view, min_span, t_hi - t_lo))
