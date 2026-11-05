import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

path = Path(__file__).resolve().parent.parent
if str(path) not in sys.path:
    sys.path.append(str(path))

from autoscale import (choose_freq_limit, choose_time_window,
                       find_spectral_lines, follow_x, log_ylim, nice_ceil,
                       padded_limits, visible_y_range)

FS = 44100


@pytest.fixture
def t():
    return np.arange(FS) / FS


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


@pytest.mark.parametrize("x, expected", [
    (550, 600), (0.013, 0.015), (1, 1), (1000, 1000), (0, 1), (-5, 1),
])
def test_nice_ceil(x, expected):
    assert nice_ceil(x) == pytest.approx(expected)


def test_padded_limits_regular():
    lo, hi = padded_limits(-1.0, 1.0, margin=0.1)
    assert (lo, hi) == pytest.approx((-1.2, 1.2))


@pytest.mark.parametrize("value", [0.0, 5.0, -3.0])
def test_padded_limits_constant_has_nonzero_height(value):
    lo, hi = padded_limits(value, value)
    assert hi > lo and lo < value < hi


def test_padded_limits_rounding_noise_is_treated_as_constant():
    lo, hi = padded_limits(5.0, 5.0 + 1e-15)
    assert hi - lo > 0.1


def test_padded_limits_symmetric():
    lo, hi = padded_limits(-1.0, 3.0, margin=0.0, symmetric=True)
    assert (lo, hi) == pytest.approx((-3.0, 3.0))


def test_padded_limits_nan_does_not_crash():
    assert padded_limits(np.nan, 1.0) == (-1.0, 1.0)


def test_log_ylim_ignores_numerical_noise_floor():
    values = np.full(1000, 1e-17)
    values[[100, 900]] = 0.5
    bottom, top = log_ylim(values)
    assert bottom >= 1e-4 * 0.5
    assert top > 0.5


def test_log_ylim_keeps_at_least_min_decades():
    bottom, top = log_ylim(np.array([1.0, 0.9]), min_decades=2.0)
    assert np.log10(top / bottom) >= 2.0


def test_log_ylim_empty_or_zero():
    assert log_ylim(np.zeros(5)) == (1e-12, 1.0)
    assert log_ylim(np.array([])) == (1e-12, 1.0)


def test_log_ylim_power_uses_10log():
    values = np.array([1.0, 1e-9, 1e-12])
    b_power, _ = log_ylim(values, dynamic_range_db=100, power=True)
    b_ampl, _ = log_ylim(values, dynamic_range_db=100, power=False)
    assert b_power < b_ampl


def _amp_spectrum(x):
    n = len(x)
    return (np.fft.fftshift(np.fft.fftfreq(n, 1 / FS)),
            np.fft.fftshift(np.abs(np.fft.fft(x)) / n))


def test_freq_limit_for_single_tone(t):
    f, a = _amp_spectrum(np.sin(2 * np.pi * 440 * t))
    limit = choose_freq_limit(f, a, FS)
    assert 440 < limit <= 1000


def test_freq_limit_for_white_noise_is_nyquist(t):
    rng = np.random.default_rng(0)
    f, a = _amp_spectrum(rng.normal(size=len(t)))
    assert choose_freq_limit(f, a, FS) == pytest.approx(FS / 2)


def test_freq_limit_zero_signal():
    f = np.fft.fftfreq(100, 1 / FS)
    assert choose_freq_limit(f, np.zeros(100), FS) == FS / 2


def test_time_window_single_tone_shows_about_five_periods(t):
    lo, hi = choose_time_window(t, np.sin(2 * np.pi * 440 * t))
    assert lo == 0.0
    assert hi == pytest.approx(5 / 440, rel=0.05)


def test_time_window_two_tones(t):
    x = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    _, hi = choose_time_window(t, x)
    assert hi == pytest.approx(5 / 440, rel=0.05)


def test_time_window_gaussian_pulse_follows_the_pulse(t):
    x = np.exp(-((t - 0.5) ** 2) / (2 * 0.001 ** 2))
    lo, hi = choose_time_window(t, x)
    assert lo < 0.5 < hi
    assert hi - lo < 0.05


def test_time_window_decaying_sine_covers_the_decay(t):
    x = np.exp(-1000 * t) * np.sin(2 * np.pi * 440 * t)
    lo, hi = choose_time_window(t, x)
    assert lo == 0.0
    assert 0.002 < hi < 0.02


def test_time_window_white_noise_shows_limited_number_of_samples(t):
    rng = np.random.default_rng(0)
    lo, hi = choose_time_window(t, rng.normal(size=len(t)))
    assert (hi - lo) * FS == pytest.approx(1000, abs=2)


def test_time_window_chirp_is_not_mistaken_for_line_spectrum():
    fs = 8000
    t = np.arange(2 * fs) / fs
    x = np.sin(2 * np.pi * (100 * t + 0.5 * (1900 / 2) * t ** 2))
    lo, hi = choose_time_window(t, x)
    assert hi - lo > 0.05


def test_time_window_short_signal_returns_whole_record():
    t = np.arange(5) / 100
    assert choose_time_window(t, np.arange(5.0)) == (0.0, t[-1])


def test_time_window_zero_signal_does_not_crash(t):
    lo, hi = choose_time_window(t, np.zeros_like(t))
    assert hi > lo


def test_find_spectral_lines_rejects_plateau():
    a = np.full(500, 1.0)
    a[[10, 400]] = 1.2
    assert find_spectral_lines(a) is None


def _step_signal():
    t = np.linspace(0, 1, 2001)
    y = np.where(t < 0.5, 1.0, 10.0) * np.sin(2 * np.pi * 20 * t)
    return t, y


def test_visible_y_range_ignores_axvline():
    t, y = _step_signal()
    fig, ax = plt.subplots()
    ax.plot(t, y)
    ax.axvline(0.1)
    ax.set_xlim(0, 0.2)
    lo, hi = visible_y_range(ax)
    assert -1.01 <= lo and hi <= 1.01


def test_follow_x_rescales_y_when_x_changes():
    t, y = _step_signal()
    fig, ax = plt.subplots()
    ax.plot(t, y)
    follow_x(ax, margin=0.05)

    ax.set_xlim(0, 0.2)
    assert max(abs(v) for v in ax.get_ylim()) < 1.5

    ax.set_xlim(0.6, 0.8)
    assert max(abs(v) for v in ax.get_ylim()) > 9.0

    ax.set_xlim(0, 0.2)
    assert max(abs(v) for v in ax.get_ylim()) < 1.5


def test_follow_x_updates_all_axes_of_shared_group():
    t, y = _step_signal()
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True)
    a1.plot(t, y)
    a2.plot(t, 3 * y)
    follow_x([a1, a2], margin=0.0)

    a1.set_xlim(0.6, 0.8)
    assert a1.get_ylim()[1] == pytest.approx(10, rel=0.02)
    assert a2.get_ylim()[1] == pytest.approx(30, rel=0.02)


def test_follow_x_respects_clip():
    t = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    ax.plot(t, -200 * t)
    follow_x(ax, margin=0.05, clip=(-120.0, None))
    assert ax.get_ylim()[0] == pytest.approx(-120.0)


def test_follow_x_with_zoom_between_samples_keeps_line_visible():
    t = np.array([0.0, 1.0, 2.0])
    fig, ax = plt.subplots()
    ax.plot(t, [0.0, 10.0, 0.0])
    follow_x(ax, margin=0.0)
    ax.set_xlim(0.2, 0.4)
    lo, hi = ax.get_ylim()
    assert lo == pytest.approx(2.0, abs=1e-6) and hi == pytest.approx(4.0, abs=1e-6)


def _plot_for(x, t, hann=False, **kw):
    from graph import Plot
    from spec import Spectr

    class Sig:
        pass

    s = Sig()
    s.t, s.data = t, x
    sp = Spectr(s, FS, hann)
    amp = sp.amplspec_shifted()
    ph = sp.phasespec_shifted()
    fr = sp.freqs_shifted()
    fig = Plot(FS, x, amp, ph, t, fr, "test", **kw).plots(show=False, save=False)
    return fig


def test_plot_sine_axes_are_autoscaled(t):
    fig = _plot_for(np.sin(2 * np.pi * 440 * t), t)
    ax = {a.get_title().split(':')[0]: a for a in fig.axes}

    amp = ax["Амплитудный спектр"]
    assert max(abs(v) for v in amp.get_xlim()) <= 1000     # не ±20000
    assert amp.get_ylim()[0] >= 1e-6                       # не 1e-17
    # фазовый график делит X со спектром амплитуд
    assert ax["Фазовый спектр"].get_xlim() == amp.get_xlim()

    sig = ax["Сигнал"]
    assert sig.get_xlim()[1] == pytest.approx(5 / 440, rel=0.05)
    assert max(abs(v) for v in sig.get_ylim()) < 1.2


def test_plot_small_amplitude_is_not_flattened(t):
    fig = _plot_for(0.02 * np.sin(2 * np.pi * 440 * t), t)
    sig = fig.axes[0]
    assert max(abs(v) for v in sig.get_ylim()) < 0.03


def test_plot_manual_override_wins(t):
    fig = _plot_for(np.sin(2 * np.pi * 440 * t), t,
                    time_window=(0.0, 0.05), freq_limit=5000)
    assert fig.axes[0].get_xlim() == pytest.approx((0.0, 0.05))
    amp = [a for a in fig.axes if a.get_title() == "Амплитудный спектр"][0]
    assert amp.get_xlim() == pytest.approx((-5000, 5000))


@pytest.mark.parametrize("phase_deg", [0, 30, 90, -45])
def test_phase_spectrum_matches_initial_phase(t, phase_deg):
    from spec import Spectr
    class Sig:
        pass

    s = Sig()
    s.t = t
    s.data = np.sin(2 * np.pi * 440 * t + np.deg2rad(phase_deg))
    sp = Spectr(s, FS)
    sp.amplspec_shifted()
    phase = sp.phasespec_shifted()
    f = sp.freqs_shifted()
    k = np.argmin(abs(f - 440))
    expected = (phase_deg - 90 + 180) % 360 - 180
    assert phase[k] == pytest.approx(expected, abs=0.01)
