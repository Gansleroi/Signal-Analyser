import numpy as np
from signals import *
from spec import Spectr
from graph import Plot
from exporter import AudioExporter

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
                    
def snap_to_bin(desired_freq: float, fs: float, N: int) -> float:
    if desired_freq == 0:
        return 0.0
    
    df = fs / N
    k = round(desired_freq / df)
    
    if k == 0:
        k = 1
        
    exact_freq = k * df
    return exact_freq


class Cli:
    
    
    def __init__(self):

        print("Вас приветствует программа анализа сложных сигналов!")
        M = get_int("Пожалуйста введите количество сигналов.\n")
        fs = get_float("Введите частоту дискретизации (Гц):\n")
        ts = get_float("Введите время реализации (с).\n")
        
        N = int(fs * ts)
        dt = 1 / fs
        
        t = np.arange(N) * dt

        signals = []

        for i in range(0, M):

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
            print("9. Гаусов импульс")
            print("10. Белый шум")
            print("11. Розовый шум (1/f)")
            print("12. Броуновский шум (1/f^2)")

            form = get_int("")
            ampl = get_float("Введите амплитуду сигнала (В)\n")
            
            freq = 0.0
            phase = 0.0
            sigma = 0.0
            
            if form not in [9, 10, 11, 12]:
                user_freq = get_float("Введите частоту сигнала (Гц)\n")
               
                freq = snap_to_bin(user_freq, fs, N) 
                
                if freq >= fs / 2:
                    print(f"ВНИМАНИЕ: Частота {freq:.1f} Гц слишком высока для Fs={fs} Гц.")
                    print(f"Максимально допустимая: {fs/2} Гц. Сигнал будет искажен.")
                
                if freq != user_freq:
                    print(f"Частота скорректирована до {freq:.2f} Гц для идеального спектра (без растекания).")
                
                phase = get_float("Введите начальную фазу сигнала в градусах.\n")
            
            else:
                sigma = get_float("Введите ширину импульса sigma (в секундах):\n")

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
                freq_mod = get_float("Введите модулирующую частоту (Гц):\n")
                mod_index = get_float("Введите коэффициент модуляции (например, 1.0 - 5.0):\n")
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
                
                case 9:
                    signal = GaussianSignal(ampl=ampl, sigma=sigma, phase=phase)
                    signals.append(signal)
                
                case 10:
                    signal = WhiteNoiseSignal(ampl=ampl)
                    signals.append(signal)
                case 11:
                    signal = PinkNoiseSignal(ampl=ampl)
                    signals.append(signal)
                case 12:
                    signal = BrownianNoiseSignal(ampl=ampl)
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
            
        print("\nГенерация и сложение сигналов...")

        for sig in signals[1:]:
            sig.generate(t)
            combined_sig = combined_sig + sig

        ans_hann = input("Применить окно Ханна для уменьшения растекания? [y / n]:\n")
        use_hann = ans_hann.strip().lower() == 'y'
        
        print("Анализ спектра...")

        spec = Spectr(signal_obj=combined_sig, fs=fs, use_hann=use_hann)
        spec_amplitude = spec.amplspec_shifted()
        spec_phase = spec.phasespec_shifted()
        freqs = spec.freqs_shifted()

        pl = Plot(
            signal=combined_sig.data,
            amplsp=spec_amplitude,
            phasesp=spec_phase,
            t=combined_sig.t,
            freqs=freqs,
            description=combined_description
        )
        ans_audio = input("Сохранить сигнал в WAV для прослушивания? [y / n]: ").strip().lower()
        if ans_audio == 'y':
            from datetime import datetime
            timestamp = datetime.now().strftime("%H-%M-%S")
            filename = f"audio_signal_{timestamp}.wav"
            
            AudioExporter.save_to_wav(filename, combined_sig.data, fs)
        
        print("Отрисовка графиков...")
        
        pl.plots()
        
        ans_emd = input("Применить Эмпирическую Модовую Декомпозицию (EMD)? [y / n]: ").strip().lower()
        if ans_emd == 'y':
            from emd import EMDAnalyzer 
            
            emd_analyzer = EMDAnalyzer(signal_obj=combined_sig)
            emd_analyzer.decompose(max_imf=4)
            emd_analyzer.plot_imfs(description=combined_description)
            
