import os
import numpy as np
from scipy.io import wavfile

class AudioExporter:
    @staticmethod
    def save_to_wav(filename: str, signal: np.ndarray, fs: float) -> None:
        max_val = np.max(np.abs(signal))
        if max_val == 0:
            print("Ошибка: сигнал пуст, нечего сохранять.")
            return

        normalized_signal = signal / max_val

        N = len(normalized_signal)
        fade_len = int(fs * 0.05)
        if fade_len > 0 and N > 2 * fade_len:
            fade_in = np.linspace(0, 1, fade_len)
            normalized_signal[:fade_len] *= fade_in
            fade_out = np.linspace(1, 0, fade_len)
            normalized_signal[-fade_len:] *= fade_out
        audio_data = np.int16(normalized_signal * 32767)

        folder_name = "audio"

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        filepath = os.path.join(folder_name, filename)

        wavfile.write(filepath, int(fs), audio_data)
        print(f"Сигнал успешно сохранен в аудиофайл: {filepath}")