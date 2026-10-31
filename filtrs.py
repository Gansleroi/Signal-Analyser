from scipy import signal
import numpy as np
import copy
import matplotlib.pyplot as plt

class ButterFil:
    def __init__(self, N: int, cut_off1: float,  btype: str, fs: float, cut_off2: float = None) -> None:
        self.N = N
        self.c1 = cut_off1
        self.c2 = cut_off2
        self.btype = btype
        self.fs = fs
        
        if self.btype in ['bandpass', 'bandstop']:
            if cut_off2 is None:
                raise ValueError("Для полосового/режекторного фильтра нужны две частоты среза (cut_off2).")
            self.Wn = (cut_off1, cut_off2)
            
        else:
            self.Wn = cut_off1
            
    def design(self):
        
        self.sos = signal.butter(
            self.N,
            self.Wn,
            btype=self.btype,
            analog=False,
            output='sos',
            fs=self.fs
        )
        
        return self.sos
    
    def apply(self, signal_obj: object) -> object:
        if not hasattr(self, 'sos'):
            self.design()
        
        data = signal_obj.data
        filtred_data = signal.sosfiltfilt(self.sos, data)
        new_signal = copy.deepcopy(signal_obj)
        new_signal.data = filtred_data
        return new_signal

    
    def plot(self):
        if not hasattr(self, 'sos'):
            self.design()
            
        w, h = signal.sosfreqz(self.sos, worN=4096, fs=self.fs)
        magnitude_db = 20 * np.log10(np.abs(h))

        plt.figure(figsize=(10, 5))
        plt.plot(w, magnitude_db)
        plt.title(f'АЧХ фильтра (Тип: {self.btype})')
        plt.xlabel('Частота, Гц')
        plt.ylabel('Усиление, дБ')
        plt.grid(True)

        if self.btype in ['bandpass', 'bandstop']:
            plt.xlim(self.Wn[0] * 0.5, self.Wn[1] * 1.5)
        else:
            plt.xlim(0, self.Wn * 2)
            
        plt.ylim(-60, 5)
        plt.axhline(-3, color='r', linestyle='--')
        plt.show()
            
        
    
    
        