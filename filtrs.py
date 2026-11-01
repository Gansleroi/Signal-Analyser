from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

class IIRFilter:
    def __init__(self, N, cut_off1, btype, fs, ftype='butter', cut_off2=None, rp=None, rs=None):
        self.N = N
        self.btype = btype
        self.fs = fs
        self.ftype = ftype
        self.rp = rp
        self.rs = rs

        if btype in ['bandpass', 'bandstop']:
            self.Wn = (cut_off1, cut_off2)
        else:
            self.Wn = cut_off1
        self.design()

    def design(self):

        kwargs = {
            'N': self.N, 'Wn': self.Wn, 'btype': self.btype, 
            'ftype': self.ftype, 'output': 'sos', 'fs': self.fs
        }

        if self.rp is not None: kwargs['rp'] = self.rp
        if self.rs is not None: kwargs['rs'] = self.rs
            
        self.sos = signal.iirfilter(**kwargs)

    def apply(self, signal_obj):
        if not hasattr(self, 'sos'):
            self.design()
        filtered_data = signal.sosfiltfilt(self.sos, signal_obj.data)
        
        import copy
        new_signal = copy.deepcopy(signal_obj)
        new_signal.data = filtered_data
        return new_signal

    def plot(self):
        w, h = signal.sosfreqz(self.sos, worN=65536, fs=self.fs)
        plt.figure(figsize=(10, 5))
        plt.plot(w, 20 * np.log10(np.abs(h)))
        plt.title(f'АЧХ: {self.ftype} ({self.btype})')
        plt.grid(True)
        plt.show()