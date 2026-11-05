import copy
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

class Filter:

    def plot(self, show: bool = True) -> None:
        if self.b_or_sos():
            w, h = signal.freqz(self.b, worN=65536, fs=self.fs)
        else:
            w, h = signal.freqz(self.sos, worN=65536, fs=self.fs)
        mag = np.abs(h)
        mag_db = 20 * np.log10(np.maximum(mag, 1e-10))

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(w, mag_db)
        plt.title(f'АЧХ')
        plt.xlabel('Частота (Гц)')
        plt.ylabel('Амплитуда (дБ)')
        plt.grid(True)
        plt.xlim(0, int(self.fs / 2))

        plt.subplot(2, 1, 2)
        plt.plot(w, np.unwrap(np.angle(h)))
        plt.title('Фазовая характеристика')
        plt.xlabel('Частота (Гц)')
        plt.ylabel('Фаза (радианы)')
        plt.grid(True)
        plt.xlim(0, int(self.fs / 2))

        plt.suptitle(self.__str__())
        plt.tight_layout()

        folder_name = "plots"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(folder_name, f"emd_{now}.png")
        plt.savefig(filename, dpi=150)
        print(f"График АЧХ и ФЧХ фильтра сохранен в файл: {filename}")

        if show:
            plt.show()

    def b_or_sos(self):
        if self.ftype == None:
            return True
        else:
            return False


class IIRFilter(Filter):

    def __init__(self, n: int, cut_off1: float, btype: str,
                    fs: float, ftype: str, cut_off2=None, rp=None, rs=None) -> None:
        self.n = n
        self.btype = btype
        self.fs = fs
        self.ftype = ftype
        self.rp = rp
        self.rs = rs
        self.sos: np.ndarray | None = None
        
        if btype in ['bandpass', 'bandstop']:
            if cut_off2 is None:
                raise ValueError("cut_off2 обязателен для"
                " btype='bandpass' или 'bandstop'")
            self.Wn = (cut_off1, cut_off2)
        else:
            self.Wn = cut_off1

        self.design()

    def __str__(self):

        puls = None
        if self.btype == 'bandpass':
            desc = f' Полосовой на {self.Wn[0]} - {self.Wn[1]}'

        elif self.btype == 'lowpass':
            desc = f' ФНЧ на {self.Wn}'

        elif self.btype == 'highpass':
            desc = f' ФВЧ на {self.Wn}'

        elif self.btype == 'bandstop':
            desc = f' Режекторный на {self.Wn}'

        if self.ftype == 'butter':
            f_type = 'Баттерворта'

        elif self.ftype == 'cheby1':
            f_type = 'Чебышёва 1'
            puls = f'Пульсации в полосе пропускания: {self.rp} Дб'

        elif self.ftype == 'cheby2':
            f_type = 'Чебышёва 2'
            puls = f'Пульсации в полосе задерживания: {self.rs} Дб'

        elif self.ftype == 'ellip':
            f_type = 'Эллиптический'
            puls = f'Пульсации в полосе пропускания: {self.rp} Дб, задерживания: {self.rs}'

        elif self.ftype == 'bessel':
            f_type = 'Бесселя'

        return f"БИХ фильтр "+ f_type + desc + '\n' + puls
    
    def design(self) -> None:

        kwargs = {
            'N': self.n, 'Wn': self.Wn, 'btype': self.btype,
            'ftype': self.ftype, 'output': 'sos', 'fs': self.fs
        }

        if self.rp is not None: kwargs['rp'] = self.rp
        if self.rs is not None: kwargs['rs'] = self.rs
            
        self.sos = signal.iirfilter(**kwargs)

    def apply(self, signal_obj: object) -> object:
        if not hasattr(self, 'sos'):
            self.design()
        filtered_data = signal.sosfiltfilt(self.sos, signal_obj.data)
        new_signal = copy.deepcopy(signal_obj)
        new_signal.data = filtered_data
        return new_signal


class FIRFilter(Filter):
     
    def __init__(self,n: int, cut_off1: float,
                 cut_off2: float, btype: str, fs: float, beta = None) -> None:
        self.numtaps = n + 1
        self.beta = beta
        self.fs = fs
        self.btype = btype
        self.b: np.ndarray | None = None
        self.ftype = None

        if btype in ['bandpass', 'bandstop']:
            if cut_off2 is None:
                raise ValueError("cut_off2 обязателен для"
                    " btype='bandpass' или 'bandstop'")
            self.Wn = (cut_off1, cut_off2)
        else:
            self.Wn = cut_off1

        self.design()

    def __str__(self):

        if self.btype == 'bandpass':
            desc = f'Полосовой на {self.Wn[0]} - {self.Wn[1]}'

        elif self.btype == 'lowpass':
            desc = f'ФНЧ на {self.Wn}'

        elif self.btype == 'highpass':
            desc = f'ФВЧ на {self.Wn}'

        elif self.btype == 'bandstop':
            desc = f'Режекторный на {self.Wn}'

        return "КИХ фильтр" + desc


    def design(self):

        self.b = signal.firwin(numtaps=self.numtaps,
                    cutoff=self.Wn,
                    pass_zero=self.btype,
                    window=('kaiser', self.beta),
                    fs=self.fs
        )


    def apply(self, signal_obj: object) -> object:
        if not hasattr(self, 'sos'):
            self.design()

        a = 1.0

        filtered_data = signal.filtfilt(self.b, a, signal_obj.data)

        new_signal = copy.deepcopy(signal_obj)
        new_signal.data = filtered_data

        return new_signal



