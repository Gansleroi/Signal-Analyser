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

def get_int_even(text: str, error ,min_val: int, max_val: int) -> int:
    while True:
        val = get_int_range(text, min_val=min_val, max_val=max_val)
        if val % 2 == 0:
            return val
        print(error)
                    
def snap_to_bin(desired_freq: float, fs: float, n: int) -> float:
    if desired_freq == 0:
        return 0.0
    
    df = fs / n
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
        self.signal_classes = {
                1: SineSignal, 2: SquareSignal, 3: TriangleSignal, 4: SawtoothSignal, 5: AMSignal, 6: ChirpSignal,
                7: PMSignal, 8: DampedSineSignal, 9: GaussianSignal, 10: WhiteNoiseSignal, 11: PinkNoiseSignal,
                12: BrownianNoiseSignal, 13: FMSignal, 14: BlueNoiseSignal, 15: VioletNoiseSignal,
                16: PerlinNoiseSignal, 17: UniformNoiseSignal, 18: ImpulseNoiseSignal, 19: ShotNoiseSignal,
                20: HumNoiseSignal, 21: CustomColorNoiseSignal
            }
            
        self.N = int(self.fs * self.ts)
        self.dt = 1 / self.fs
            
        self.t = np.arange(self.N) * self.dt
        
        combined_sig, description = self._create_combined_signal()  #Генерация сигналов
        
        ans_hann = get_choice("Применить окно Ханна для уменьшения растекания? [y / n]:\n", ['y', 'n'])
        self.use_hann = ans_hann == 'y'
        
        self._analyse_and_plot(signal_obj=combined_sig, description=description)  #Анализ и построение графиков
        
        self._audio_export(signal_obj=combined_sig) #Создание .waw файла
        
        self._plot_psd(signal_obj=combined_sig)  #Построение СПМ
        
        self._apply_emd(signal_obj=combined_sig, description=description)  #Применение ЭМД
        
        self._handle_filtering(signal_obj=combined_sig, description=description)  #Проектирование и применение фильтра
        
    
    def _create_combined_signal(self):  #Генерация сигналов
    
        signals = []
    
        for i in range(0, self.M):
    
            print(f"Создаётся {i + 1}-ый сигнал!")
            print("Выберите форму сигнала:")
            print("1. Синус")
            print("2. Меандр")
            print("3. Треугольный сигнал")
            print("4. Пилообразный сигнал")
            print("5. АМ-сигнал (Амплитудная модуляция)")
            print("6. ЛЧМ (Линейно-частотная модуляция)")
            print("7. ФМ (Фазовая модуляция)")
            print("8. Затухающий синус")
            print("9. Гауссов импульс")
            print("10. Белый шум")
            print("11. Розовый шум (1/f)")
            print("12. Броуновский шум (1/f^2)")
            print("13. ЧМ (Частотная модуляция")
            print("14. Синий шум (f)")
            print("15. Фиолетовый шум (f^2)")
            print("16. Шум Перлина")
            print("17. Равномерный шум")
            print("18. Импульсный шум")
            print("19. Дробовой шум")
            print("20. Сетевая наводка")
            print("21. Шум с произвольным показателем beta (спектр 1/f^beta)")

            params = {}
    
            form = get_int_range("", 1, 21)
            ampl = get_float_positive("Введите амплитуду сигнала (В)\n")
            params['ampl'] = ampl

            if form not in [9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21]:
                user_freq = get_float_positive("Введите частоту сигнала (Гц)\n")

                freq = snap_to_bin(desired_freq=user_freq, fs=self.fs, n=self.N)
                params['freq'] = freq

                if freq >= self.fs / 2:
                    print(f"ВНИМАНИЕ: Частота {freq:.1f} Гц слишком высока для Fs={self.fs} Гц.")
                    print(f"Максимально допустимая: {self.fs/2} Гц. Сигнал будет искажен.")

                if freq != user_freq:
                    print(f"Частота скорректирована до {freq:.2f} Гц для идеального спектра (без растекания).")

                phase = get_float("Введите начальную фазу сигнала в градусах.\n")
                params['phase'] = phase

            if form == 16:
                user_freq = get_float_positive("Введите частоту сигнала (Гц)\n")

                freq = snap_to_bin(desired_freq=user_freq, fs=self.fs, n=self.N)
                params['freq'] = freq

            if form == 9:
                sigma = get_float_positive("Введите ширину импульса sigma (в секундах):\n")
                params['sigma'] = sigma

            if form == 5:
                freq_mod = get_float_positive("Введите модулирующую частоту (Гц):\n")
                mod_index = get_float_range("Введите коэффициент модуляции (от 0.0 до 1.0): \n",
                                            0.0, 1.0)
                params['freq_mod'] = freq_mod
                params['mod_index'] = mod_index

            elif form == 6:
                freq_end = get_float_positive("Введите конечную частоту ЛЧМ (Гц):\n")
                params['freq_end'] = freq_end

            elif form == 7:
                freq_mod = get_float_positive("Введите модулирующую частоту (Гц):\n")
                mod_index = get_float_positive("Введите коэффициент модуляции (например, 1.0 - 5.0):\n")

                params['freq_mod'] = freq_mod
                params['mod_index'] = mod_index

            elif form == 8:
                damp_index = get_float_positive("Введите коэффициент затухания.\n")

                params['damp_index'] = damp_index

            elif form == 13:
                freq_car = get_float_positive("Введите частоту несущей.\n")
                ampl_car = get_float_positive("Введите амплитуду несущей.\n")
                freq_dev = get_float_positive("Введите девиацию частоты\n")

                params['freq_car'] = freq_car
                params['ampl_car'] = ampl_car
                params['freq_dev'] = freq_dev

            elif form == 21:
                beta = get_float_range('Введите показатель beta (Спектр шума 1/f^beta):\n')
                params['beta'] = beta

            signal = self.signal_classes[form](**params)
            signals.append(signal)
    
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
            fs=self.fs,
            signal=signal_obj.data,
            amplsp=spec_amplitude,
            phasesp=spec_phase,
            t=signal_obj.t,
            freqs=freqs,
            description=description
        )
        
        pl.plots()
        
    def _handle_filtering(self, signal_obj: object, description: str) -> None: #Проектирование и применение фильтра
        ans_filt = get_choice("Применить фильтрацию к сигналу? [y / n]\n", ['y', 'n'])
        if ans_filt == 'y':

            nyq = self.fs / 2

            ans_filter = get_choice("Какой тип фильтра использовать? (IIR(БИХ), FIR(КИХ)]\n", ['iir', 'fir'])

            btype = get_choice(
                "Какой фильтр нужен (lowpass, highpass, bandpass, bandstop)? Введите название на английском\n",
                ['lowpass', 'highpass', 'bandpass', 'bandstop']
            )

            order = 0

            if btype == 'highpass' or btype == 'bandstop' and ans_filter == 'fir':
                order = get_int_even('Введите порядок фильтра\n',
                                     "Для такого типа фильтра порядок должен быть чётным!",
                                     0, 100000)

            elif btype == 'lowpass' or btype == 'bandpass' and ans_filter == 'fir':
                order = get_int_range("Введите порядок фильтра\n", 1, 100000)

            elif ans_filter == 'iir':
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

            if ans_filter == 'iir':
                ftype = get_choice(
                    "Выберите тип фильтра (butter, cheby1, cheby2, ellip, bessel):\n",
                    ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
                )
                

                if ftype in ['cheby1', 'ellip']:
                    rp = get_float_range("Введите пульсации в полосе пропускания (дБ, обычно 0.5 - 3):\n",
                                         0.01, 10.0)
                
                if ftype in ['cheby2', 'ellip']:
                    rs = get_float_range("Введите пульсации в полосе задерживания (дБ, обычно 40 - 80):\n",
                                         1.0, 150.0)

                from filters import IIRFilter

                filter = IIRFilter(
                    n=order,
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

                beta = get_float_range("Параметр формы окна Кайзера (beta)" 
                "Чем больше beta, тем ниже боковые лепестки и лучше подавление," 
                " но шире переходная полоса.", 0, 20)

                filter = FIRFilter(
                    n=order,
                    cut_off1=cut_off1,
                    cut_off2=cut_off2,
                    btype=btype,
                    fs=self.fs,
                    beta=beta
                )

            del ans_filter
                    
            ans_plot = get_choice("Построть АЧХ фильтра? [y / n]\n", ['y', 'n'])
            if ans_plot == 'y':
                filter.plot()

            del ans_plot
            print("Применение фильтра...")
                    
            filtered_sig = filter.apply(signal_obj)
            
            self._analyse_and_plot(filtered_sig, (description + ' После фильтра'))
            
            self._audio_export(signal_obj=filtered_sig)
        del ans_filt
    def _apply_emd(self, signal_obj: object, description: str) -> None:  #Применение ЭМД
        ans_emd = get_choice("Применить Эмпирическую Модовую Декомпозицию (EMD)? [y / n]:\n", ['y', 'n'])
        if ans_emd == 'y':
            from emd import EMDAnalyzer
            emd_analyzer = EMDAnalyzer(signal_obj=signal_obj)
            emd_analyzer.decompose(max_imf=4)
            emd_analyzer.plot_imfs(description=description)
            emd_analyzer.plot_imf_spectra()

    def _audio_export(self, signal_obj: object) -> object: #Аудио экспорт
        ans_audio = get_choice("Сохранить сигнал в WAV для прослушивания? [y / n]:\n", ['y', 'n'])
        if ans_audio == 'y':
            from datetime import datetime
            from exporter import AudioExporter
            timestamp = datetime.now().strftime("%H-%M-%S")
            filename = f"audio_signal_{timestamp}.wav"
        
            AudioExporter.save_to_wav(filename, signal_obj.data, self.fs)
    
    def _plot_psd(self, signal_obj: object) -> object:
        ans_psd = get_choice("Построить СПМ сигнала? [y / n]\n", ['y', 'n'])
        if ans_psd == 'y':
            from PSD import PSD
            psd = PSD(fs=self.fs, sig=signal_obj.data)
            psd.welch()
    