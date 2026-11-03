import numpy as np
from signals import *
import sys

def get_float(text: str):
    while True:
        try:
            return float(input(text))
        except ValueError:
            print("Ошибка! Введите корректное числовое значение.\n")

def get_float_positive(text: str) -> float:
    while True:
        val = get_float(text)
        if val > 0:
            return val
        print("Ошибка! Значение должно быть больше нуля.\n")

def get_float_range(text: str, min_val: float, max_val: float) -> float:
    while True:
        val = get_float(text)
        if min_val <= val <= max_val:
            return val
        print(f"Ошибка! Значение должно быть в диапазоне [{min_val}, {max_val}].\n")

def get_int(text: str) -> int:
    while True:
        try:
            return int(input(text))
        except ValueError:
            print("Ошибка! Введите целое число.\n")

def get_int_range(text: str, min_val: int, max_val: int) -> int:
    while True:
        val = get_int(text)
        if min_val <= val <= max_val:
            return val
        print(f"Ошибка! Введите число от {min_val} до {max_val}.\n")

def get_choice(text: str, valid: list) -> str:
    while True:
        val = input(text).strip().lower()
        if val in valid:
            return val
        print(f"Ошибка! Допустимые варианты: {', '.join(valid)}.\n")
                    
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
        self.M = get_int_range("Пожалуйста введите количество сигналов.\n", 1, 100)
        self.fs = get_float_positive("Введите частоту дискретизации (Гц):\n")
        self.ts = get_float_positive("Введите время реализации (с).\n")
            
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
    
            form = get_int_range("", 1, 12)
            ampl = get_float_positive("Введите амплитуду сигнала (В)\n")

            freq = 0.0
            phase = 0.0
            sigma = 0.0

            if form not in [9, 10, 11, 12]:
                user_freq = get_float_positive("Введите частоту сигнала (Гц)\n")

                freq = snap_to_bin(user_freq, self.fs, self.N)

                if freq >= self.fs / 2:
                    print(f"ВНИМАНИЕ: Частота {freq:.1f} Гц слишком высока для Fs={self.fs} Гц.")
                    print(f"Максимально допустимая: {self.fs/2} Гц. Сигнал будет искажен.")

                if freq != user_freq:
                    print(f"Частота скорректирована до {freq:.2f} Гц для идеального спектра (без растекания).")

                phase = get_float("Введите начальную фазу сигнала в градусах.\n")

            if form == 9:
                sigma = get_float_positive("Введите ширину импульса sigma (в секундах):\n")

            freq_mod = 0.0
            mod_index = 0.0
            freq_end = 0.0
            damp_index = 0.0

            if form == 5:
                freq_mod = get_float_positive("Введите модулирующую частоту (Гц):\n")
                mod_index = get_float_range("Введите коэффициент модуляции (от 0.0 до 1.0): \n", 0.0, 1.0)
            elif form == 6:
                freq_end = get_float_positive("Введите КОНЕЧНУЮ частоту ЛЧМ (Гц):\n")
            elif form == 7:
                freq_mod = get_float_positive("Введите модулирующую частоту (Гц):\n")
                mod_index = get_float_positive("Введите коэффициент модуляции (например, 1.0 - 5.0):\n")
            elif form == 8:
                damp_index = get_float_positive("Введите коэффициент затухания.\n")
    
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
        ans_filt = input("Применить фильтрацию к сигналу? [y / n]\n").strip().lower()
        if ans_filt == 'y':

            

            nyq = self.fs / 2

            btype = get_choice(
                "Какой фильтр нужен (lowpass, highpass, bandpass, bandstop)? Введите название на английском\n",
                ['lowpass', 'highpass', 'bandpass', 'bandstop']
            )

            order = get_int_range("Введите порядок фильтра\n", 1, 100000)
            
            cut_off1 = get_float_range("Введите частоту среза (Гц)\n", 0.1, nyq - 0.1)

            cut_off2 = None

            if btype in ['bandpass', 'bandstop']:
                cut_off2 = get_float_range(
                    f"Введите вторую частоту среза (Гц, > {cut_off1})\n",
                    cut_off1 + 0.1, nyq - 0.1
                )

            rp = None
            rs = None

            ans_filter = get_choice("Какой тип фильтра использовать? (IIR(БИХ), FIR(КИХ)]\n", ['iir', 'fir'])
            if ans_filter == 'iir':
                ftype = get_choice(
                    "Выберите тип фильтра (butter, cheby1, cheby2, ellip, bessel):\n",
                    ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
                )
                

                if ftype in ['cheby1', 'ellip']:
                    rp = get_float_range("Введите пульсации в полосе пропускания (дБ, обычно 0.5 - 3):\n", 0.01, 10.0)
                
                if ftype in ['cheby2', 'ellip']:
                    rs = get_float_range("Введите затухание в полосе задерживания (дБ, обычно 40 - 80):\n", 1.0, 150.0)

                from filters import IIRFilter

                filter = IIRFilter(
                    N=order,
                    cut_off1=cut_off1,
                    cut_off2=cut_off2,
                    btype=btype,
                    fs=self.fs,
                    ftype=ftype,
                    rp=rp,
                    rs=rs
                )

            elif ans_filter == 'fir':

                from filters import FIRFilter

                beta = get_float_range("Параметр формы окна Кайзера (beta)" \
                "Чем больше beta, тем ниже боковые лепестки и лучше подавление," \
                " но шире переходная полоса.", 0, 20)

                filter = FIRFilter(
                    N=order,
                    cut_off1=cut_off1,
                    cut_off2=cut_off2,
                    btype=btype,
                    fs=self.fs,
                    beta=beta
                )
                    
            ans_plot = get_choice("Построть АЧХ фильтра? [y / n]\n", ['y', 'n'])
            if ans_plot == 'y':
                filter.plot()
                    
            print("Применение фильтра...")
                    
            filtered_sig = filter.apply(signal_obj)
            
            self._analyse_and_plot(filtered_sig, (description + ' После фильтра'))
            
            self._audio_export(signal_obj=filtered_sig)
            
    def _apply_EMD(self, signal_obj: object, description: str) -> None:  #Применение ЭМД
        ans_emd = get_choice("Применить Эмпирическую Модовую Декомпозицию (EMD)? [y / n]:\n", ['y', 'n'])
        if ans_emd == 'y':
            from emd import EMDAnalyzer 
            emd_analyzer = EMDAnalyzer(signal_obj=signal_obj)
            emd_analyzer.decompose(max_imf=4)
            emd_analyzer.plot_imfs(description=description)

    def _audio_export(self, signal_obj: object) -> object: #Аудио экспорт
        ans_audio = get_choice("Сохранить сигнал в WAV для прослушивания? [y / n]:\n", ['y', 'n'])
        if ans_audio == 'y':
            from datetime import datetime
            from exporter import AudioExporter
            timestamp = datetime.now().strftime("%H-%M-%S")
            filename = f"audio_signal_{timestamp}.wav"
        
            AudioExporter.save_to_wav(filename, signal_obj.data, self.fs)
    
    def _plot_PSD(self, signal_obj: object) -> object:
        ans_PSD = get_choice("Построить СПМ сигнала? [y / n]\n", ['y', 'n'])
        if ans_PSD == 'y':
            from PSD import PSD
            psd = PSD(fs=self.fs, signal=signal_obj.data)
            psd.welch()
    