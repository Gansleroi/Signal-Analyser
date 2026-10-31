import numpy as np
from signals import *
import sys

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
        self.M = get_int("Пожалуйста введите количество сигналов.\n")
        self.fs = get_float("Введите частоту дискретизации (Гц):\n")
        self.ts = get_float("Введите время реализации (с).\n")
            
        self.N = int(self.fs * self.ts)
        self.dt = 1 / self.fs
            
        self.t = np.arange(self.N) * self.dt
        
        combined_sig, description = self._create_combined_signal()  #Генерация сигналов
        
        ans_hann = input("Применить окно Ханна для уменьшения растекания? [y / n]:\n")
        self.use_hann = ans_hann.strip().lower() == 'y'
        
        self._analyse_and_plot(signal_obj=combined_sig, description=description)  #Анализ и построение графиков
        
        self._audio_export(signal_obj=combined_sig) #Создание .waw файла
        
        self._plot_PSD(signal_obj=combined_sig)  #Построение СПМ
        
        self._apply_EMD(signal_obj=combined_sig, description=description)  #Применение ЭМД
        
        self._handle_filtering(signal_obj=combined_sig, description=description)  #Проектирование и применение фильтра
        
    
    def _create_combined_signal(self):  #Генерация сигналов
    
        signals = []
    
        for i in range(0, self.M):
    
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
                   
                freq = snap_to_bin(user_freq, self.fs, self.N) 
                    
                if freq >= self.fs / 2:
                    print(f"ВНИМАНИЕ: Частота {freq:.1f} Гц слишком высока для Fs={self.fs} Гц.")
                    print(f"Максимально допустимая: {self.fs/2} Гц. Сигнал будет искажен.")
                    
                if freq != user_freq:
                    print(f"Частота скорректирована до {freq:.2f} Гц для идеального спектра (без растекания).")
                    
                phase = get_float("Введите начальную фазу сигнала в градусах.\n")
                
            if form == 9:
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
            print("Список сигналов пуст! Завершение программы...")
            sys.exit()
    
        descriptions = [str(s) for s in signals]
        combined_description = " + ".join(descriptions)
    
        combined_sig = signals[0]
        combined_sig.generate(self.t)
                
        print("\nГенерация и сложение сигналов...")
    
        for sig in signals[1:]:
            sig.generate(self.t)
            combined_sig = combined_sig + sig
        
        return combined_sig, combined_description
    
    def _analyse_and_plot(self, signal_obj: object, description: str) -> None: #Анализ спектра и построение графиков
        from spec import Spectr
        from graph import Plot
        
        spec = Spectr(signal_obj=signal_obj, fs=self.fs, use_hann=self.use_hann)
        spec_amplitude = spec.amplspec_shifted()
        spec_phase = spec.phasespec_shifted()
        freqs = spec.freqs_shifted()
        
        pl = Plot(
            signal=signal_obj.data,
            amplsp=spec_amplitude,
            phasesp=spec_phase,
            t=signal_obj.t,
            freqs=freqs,
            description=description
        )
        
        pl.plots()
        
    def _handle_filtering(self, signal_obj: object, description: str) -> None: #Проектирование и применение фильтра
        ans_filt = input("Применить фильтрацию к сигналу? [y / n]\n")
        if ans_filt == 'y':
            from filtrs import ButterFil
            order = get_int("Введите порядок фильтра\n")
            btype = input("Какой фильтр нужен (lowpass, highpass, bandpass, bandstop)? Введите название на английском\n")
            cut_off1 = get_float("Введите частоту среза\n")
            if btype in ['bandpass', 'bandstop']:
                cut_off2 = get_float("Введите вторую частоту среза\n")
                filtr = ButterFil(
                    N = order,
                    cut_off1 = cut_off1,
                    cut_off2 = cut_off2,
                    btype = btype,
                    fs = self.fs
                )
                    
            else:
                filtr = ButterFil(
                    N = order,
                    cut_off1 = cut_off1,
                    cut_off2 = None,
                    btype = btype,
                    fs = self.fs
                )
                    
            ans_plot = input("Построть АЧХ фильтра? [y / n]\n")
            if ans_plot == 'y':
                filtr.plot()
                    
            print("Применение фильтра...")
                    
            filtred_sig = filtr.apply(signal_obj)
            
            self._analyse_and_plot(filtred_sig, (description+' После фильтра'))
            
    def _apply_EMD(self, signal_obj: object, description: str) -> None:  #Применение ЭМД
        ans_emd = input("Применить Эмпирическую Модовую Декомпозицию (EMD)? [y / n]:\n").strip().lower()
        if ans_emd == 'y':
            from emd import EMDAnalyzer 
            emd_analyzer = EMDAnalyzer(signal_obj=signal_obj)
            emd_analyzer.decompose(max_imf=4)
            emd_analyzer.plot_imfs(description=description)

    def _audio_export(self, signal_obj: object) -> object: #Аудио экспорт
        ans_audio = input("Сохранить сигнал в WAV для прослушивания? [y / n]:\n").strip().lower()
        if ans_audio == 'y':
            from datetime import datetime
            from exporter import AudioExporter
            timestamp = datetime.now().strftime("%H-%M-%S")
            filename = f"audio_signal_{timestamp}.wav"
        
            AudioExporter.save_to_wav(filename, signal_obj.data, self.fs)
    
    def _plot_PSD(self, signal_obj: object) -> object:
        ans_PSD = input("Построить СПМ сигнала? [y / n]\n").strip().lower()
        if ans_PSD == 'y':
            from PSD import PSD
            psd = PSD(fs=self.fs, signal=signal_obj.data)
            psd.welch()
    