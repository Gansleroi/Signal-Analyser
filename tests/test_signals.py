import pytest
from pathlib import Path
import sys

path = Path(__file__).resolve().parent.parent
if str(path) not in sys.path:
    sys.path.append(str(path))

from signals import *

Fs = 44100

@pytest.fixture
def t():
    return np.arange(Fs) / Fs

@pytest.fixture
def t_half():
    return (np.arange(Fs) + 0.5) / Fs

#Амплитуда во временной области
AMPLS = [1.0, 4.0, 10.0]
FREQS = [100.0, 200.0, 400.0, 1000.0]

@pytest.mark.parametrize("cls, tol", [
    (SineSignal,     lambda a, f: a * (1 - np.cos(np.pi * f / Fs))),
    (SquareSignal,   lambda a, f: 0.0),
    (TriangleSignal, lambda a, f: 2 * a * f / Fs),
    (SawtoothSignal, lambda a, f: 2 * a * f / Fs)
])
@pytest.mark.parametrize("ampl", AMPLS)
@pytest.mark.parametrize("freq", FREQS)
def test_timeampl(t, cls, tol, ampl, freq):
    s = cls(freq=freq, ampl=ampl, phase=0.0)
    s.generate(t)
    eps = tol(ampl, freq) + 1e-9
    assert np.max(s.data) == pytest.approx(ampl, abs=eps)
    assert np.min(s.data) == pytest.approx(-ampl, abs=eps)                


#Амлитуда в частотной области
@pytest.mark.parametrize("freq", FREQS)
def test_sine_frequency(t, freq):

    s = SineSignal(freq=freq, ampl=4.0, phase=0.0)
    s.generate(t=t)

    spec = abs(np.fft.rfft(s.data)) / len(t) * 2
    freqs = np.fft.rfftfreq(len(t), 1 / Fs)

    assert freqs[np.argmax(spec)] == pytest.approx(freq, abs=1.0)
    assert spec.max() == pytest.approx(4.0, abs=1e-3)

@pytest.mark.parametrize("freq", FREQS)
def test_square_frequency(t, freq):

    s = SquareSignal(freq=freq, ampl=4.0, phase=0.0)
    s.generate(t=t)

    spec = abs(np.fft.rfft(s.data)) / len(t) * 2
    freqs = np.fft.rfftfreq(len(t), 1 / Fs)

    assert freqs[np.argmax(spec)] == pytest.approx(freq, abs=1.0)
    #Амплитуда n-ой нечётной гармоники изменяется по формуле: An = (4 * A) / (pi * n), где n - номер нечётной гармоники
    #Амплитуда чётных гармоник равна нулю
    k_f = int(freq)
    assert spec[0] < 0.02 * spec[k_f]
    assert spec[k_f] == pytest.approx((4 * 4.0 / np.pi), rel=1e-2)
    assert spec[3*k_f] == pytest.approx((4 * 4.0) / (3 * np.pi), rel=2e-2)
    assert np.all(spec[2*k_f : 10*k_f : 2*k_f] < 0.02 * spec[k_f])

@pytest.mark.parametrize("freq", FREQS)
def test_triangle_frequency(t, freq):

    s = TriangleSignal(freq=freq, ampl=4.0, phase=0.0)
    s.generate(t=t)

    spec = abs(np.fft.rfft(s.data)) / len(t) * 2
    freqs = np.fft.rfftfreq(len(t), 1 / Fs)

    assert freqs[np.argmax(spec)] == pytest.approx(freq, abs=1.0)
    #Амплитуда n-ой нечётной гармоники изменяется по формуле: An = (8 * A) / (pi**2 * n**2), где n - номер нечётной гармоники
    #Амплитуда чётных гармоник равна нулю
    k_f = int(freq)
    assert spec[0] < 0.02 * spec[k_f]
    assert spec[k_f] == pytest.approx(((8 * 4.0) / (1 * np.pi)**2), abs=1e-3)
    assert spec[3*k_f] == pytest.approx((8 * 4.0) / (3 * np.pi)**2, abs=1e-3)
    assert np.all(spec[2*k_f : 10*k_f : 2*k_f] < 0.02 * spec[k_f])

@pytest.mark.parametrize("freq", FREQS)
def test_sawtooth_frequency(t, freq):
    s = SawtoothSignal(freq=freq, ampl=4.0, phase=0.0)
    s.generate(t)

    spec = abs(np.fft.rfft(s.data)) / len(t) * 2
    freqs = np.fft.rfftfreq(len(t), 1 / Fs)

    assert freqs[np.argmax(spec)] == pytest.approx(freq, abs=1.0)
    #Амплитуда n-ой гармоники вычисляется по формуле: An = 2 * A / pi / n , где n - номер гармоники
    k_f = int(freq)
    assert spec[0] < 0.02 * spec[k_f]
    assert spec[k_f] == pytest.approx((2 * 4.0 / np.pi), abs=1e-3)
    assert spec[2*k_f] == pytest.approx((4.0 / np.pi), abs=1e-3)
    assert spec[3*k_f] == pytest.approx((2 * 4.0 / np.pi / 3), abs=1e-3)
    assert spec[4*k_f] == pytest.approx((2 * 4.0 / np.pi / 4), abs=1e-3)


#Проверка начальной фазы
@pytest.mark.parametrize("phase", [0.0, 30.0, 45.0, 60.0, 90.0, 180.0])
def test_sine_phase_values(t, phase):
    s = SineSignal(freq=440.0, ampl=4.0, phase=phase)
    s.generate(t)
    assert s.data[0] == pytest.approx(4.0 * np.sin(np.deg2rad(phase)), abs=1e-9)

@pytest.mark.parametrize("phase, first", [(0.0, 0.0), (90.0, 4.0)])
def test_triangle_phase_values(t, phase, first):

    s = TriangleSignal(freq=440.0, ampl=4.0, phase=phase)
    s.generate(t)
    assert s.data[0] == pytest.approx(first, abs=1e-9)

@pytest.mark.parametrize("phase, first", [(0.0, 4.0), (90.0, 4.0), (180.0, -4.0), (270.0, -4.0)])
def test_square_phase_values(t, phase, first):

    s = SquareSignal(freq=440.0, ampl=4.0, phase=phase)
    s.generate(t)
    assert s.data[0] == pytest.approx(first, abs=1e-9)

@pytest.mark.parametrize("phase, first", [(0.0, -4.0), (90.0, -2.0), (180.0, .0)])
def test_sawtooth_phase_values(t, phase, first):
    s = SawtoothSignal(freq=440.0, ampl=4.0, phase=phase)
    s.generate(t)
    assert s.data[0] == pytest.approx(first, abs=1e-9)

CLASSES = [SineSignal, TriangleSignal, SquareSignal, SawtoothSignal]

#Проверка что сдвиг фазы павен сдвигу сигнала по времени
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("phase_deg", [45.0, 90.0, 180.0, 270.0])
def test_phase_is_time_shift(cls, phase_deg, t_half):
    freq = Fs / 80
    t = t_half

    s0 = cls(freq=freq, ampl=4.0, phase=0.0)
    s1 = cls(freq=freq, ampl=4.0, phase=phase_deg)
    s0.generate(t=t)
    s1.generate(t=t)

    shift = int(round(phase_deg / 360 * Fs / freq))
    assert np.allclose(s1.data[:-shift], s0.data[shift:], atol=1e-9)

#Тест длины и конечности
@pytest.mark.parametrize("cls", CLASSES)
def test_length_and_finite(t, cls):
    s = cls(freq=440.0, ampl=4.0, phase=0.0)
    s.generate(t)
    assert len(s.data) == len(t)
    assert np.all(np.isfinite(s.data))

#Проверяется что у меандра только два уровня
def test_square_only_two_levels(t):
    s = SquareSignal(freq=440.0, ampl=4.0, phase=0.0)
    s.generate(t)
    i = np.where(s.data >= 0, 1, -1)
    assert set(np.unique(s.data[i])) <= {-4.0, 4.0}

#Сравнение сгенерированных сигналов с эталонными формами
def test_sine_matches_reference(t_half):
    s1 = SineSignal(freq=440, ampl=4.0, phase=0.0)
    s1.generate(t_half)

    s2 = 4.0 * np.cos(2 * np.pi * 440 * t_half - np.pi / 2)

    assert s1.data == pytest.approx(s2, abs=1e-9)

def test_triangle_matches_reference(t_half):
    s1 = TriangleSignal(freq=440, ampl=4.0, phase=0.0)
    s1.generate(t_half)

    s2 = 4.0 * (4 * np.abs((440 * t_half - 0.25) % 1 - 0.5) - 1)

    assert s1.data == pytest.approx(s2, abs=1e-9)

def test_square_matches_reference(t_half):
    s1 = SquareSignal(freq=440, ampl=4.0, phase=0.0)
    s1.generate(t_half)

    phase = (2 * np.pi * 440 * t_half) % (2 * np.pi)

    expected = np.where(phase < np.pi, 4.0, -4.0)

    distance = np.minimum(
    phase,
    np.minimum(
        np.abs(phase - np.pi),
        2 * np.pi - phase
    )) 

    mask = distance > 1e-10

    assert s1.data[mask] == pytest.approx(
        expected[mask],
        abs=1e-9
    )

def test_sawtooth_matches_reference(t_half):
    s = SawtoothSignal(freq=440, ampl=4.0, phase=0.0)
    s.generate(t_half)

    cycles = 440 * t_half
    expected = 2 * 4.0 * (cycles - np.floor(cycles)) - 4.0

    assert s.data == pytest.approx(expected, abs=1e-9)

#Проверяется разрешение по частоте
def test_frequency_resolution():
    n = Fs

    freqs = np.fft.rfftfreq(n, 1 / Fs)

    assert freqs[1] - freqs[0] == pytest.approx(1.0)
    assert freqs[-1] == pytest.approx(Fs / 2)


def test_FM_signal():

    s = FMSignal(
        ampl=4.0,
        freq=440.0,
        phase=0.0,
        ampl_car=8.0,
        freq_car=880.0,
        freq_dev=1e3
    )

    assert s.data != 0
    