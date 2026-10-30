import numpy as np
from signals import *
from spec import Spectr
from graph import Plot

def get_float(text: str):
        while True:
            try:
                return float(input(text))
            except ValueError:
                print("Ошибка! Введите корректное значение\n")
                
def get_int(text: str):
        while True:
            try:
                return int(input(text))
            except ValueError:
                    print("Ошибка! Введите корректное значение\n")


class Cli:
    
    
    def __init__(self):

        print("Вас приветствует программа анализа сложных сигналов!")
        N = get_int("Пожалуйста введите количество сигналов.\n")
        ts = get_float("Введите время реализации (с).\n")
        t = np.linspace(0, ts, 5000)

        signals = []

        for i in range(0, N):

            print(f"Создаётся {i + 1}-ый сигнал!")
            print("Выберите форму сигнала:")
            print("1. Синус.")
            print("2. Меандр.")
            print("3. Треугольный сигнал.")
            print("4. Пилообразный сигнал.")
            print("5. АМ-сигнал (Амплитудная модуляция)")
            print("6. ЛЧМ (Линейно-частотная модуляция)")
            print("7. ФМ (Фазовая модуляция)")
            print("8. Затухающий синус")

            form = get_int("")
            freq = get_float("Введите частоту сигнала (Гц)\n")
            ampl = get_float("Введите амплитуду сигнала (В)\n")
            phase = get_float("Введите начальную фазу сигнала в градусах.\n")

            freq_mod = 0.0
            mod_index = 0.0
            freq_end = 0.0
            damp_index = 0.0
            
            if form == 5:
                freq_mod = get_float("Введите модулирующую частоту (Гц):\n")
                mod_index = get_float("Введите коэффициент модуляции (от 0.0 до 1.0): \n")
            elif form == 6:
                freq_end = get_float("Введите КОНЕЧНУЮ частоту ЛЧМ (Гц):\n")
            elif form == 7:
                freq_mod = get_float("Введите модулирующую частоту (Гц): ")
                mod_index = get_float("Введите коэффициент модуляции (например, 1.0 - 5.0): ")
            elif form == 8:
                damp_index = get_float("Введите коэффициент затухания.\n")

            match form:
                case 1:
                    signal = SineSignal(freq=freq, ampl=ampl, phase=phase)
                    signals.append(signal)

                case 2:
                    signal = SquareSignal(freq=freq, ampl=ampl, phase=phase)
                    signals.append(signal)

                case 3:
                    signal = TriangleSignal(freq=freq, ampl=ampl, phase=phase)
                    signals.append(signal)

                case 4:
                    signal = SawtoothSignal(freq=freq, ampl=ampl, phase=phase)
                    signals.append(signal)

                case 5:
                    signal = AMSignal(freq=freq, ampl=ampl, freq_mod=freq_mod, mod_index=mod_index, phase=phase)
                    signals.append(signal)
                    
                case 6:
                    signal = ChirpSignal(freq_start=freq, freq_end=freq_end, ampl=ampl, phase=phase)
                    signals.append(signal)
                case 7:
                    signal = PMSignal(freq=freq, ampl=ampl, freq_mod=freq_mod, mod_index=mod_index, phase=phase)
                    signals.append(signal)
                
                case 8:
                    signal = DampedSineSignal(freq=freq, ampl=ampl, damping=damp_index, phase=phase)
                    signals.append(signal)

                case _:
                    print("Неверный выбор! Сигнал пропущен.")

        if not signals:
            print("Список сигнало пуст! Завершение программы...")
            return

        descriptions = [str(s) for s in signals]
        combined_description = " + ".join(descriptions)

        combined_sig = signals[0]
        combined_sig.generate(t)

        ans = input("Добавить белый шум? [y / n]\n")
        if ans.strip().lower() == 'y':
            sigma = get_float("Введите СКО шума.\n")
            combined_sig.data += np.random.normal(0, sigma, len(t))
            
        print("\nГенерация и сложение сигналов...")

        for sig in signals[1:]:
            sig.generate(t)
            combined_sig = combined_sig + sig

        print("Анализ спектра...")

        ans_hann = input("Применить окно Ханна для уменьшения растекания? [y / n]:\n")
        use_hann = ans_hann.strip().lower() == 'y'

        spec = Spectr(signal_obj=combined_sig, use_hann=use_hann)
        spec_amplitude = spec.amplspec_shifted()
        spec_phase = spec.phasespec_shifted()
        freqs = spec.freqs_shifted()

        print("Отрисовка графиков...")
        pl = Plot(
            signal=combined_sig.data,
            amplsp=spec_amplitude,
            phasesp=spec_phase,
            t=combined_sig.t,
            freqs=freqs,
            description=combined_description
        )
        pl.plots()
