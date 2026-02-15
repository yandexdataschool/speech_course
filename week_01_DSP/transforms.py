from functools import partial

import librosa
import numpy as np
import scipy


class Sequential:
    def __init__(self, *args):
        self.transforms = args

    def __call__(self, inp: np.ndarray):
        res = inp
        for transform in self.transforms:
            res = transform(res)
        return res


class Windowing:
    def __init__(self, window_size=1024, hop_length=None):
        self.window_size = window_size
        self.hop_length = hop_length if hop_length else self.window_size // 2
    
    def __call__(self, waveform):
        # left_padding = np.zeros(self.window_size // 2)
        # right_padding = np.zeros(self.window_size // 2)
        # effective_waveform = np.concat([left_padding, waveform, right_padding], axis=-1)

        effective_waveform = np.pad(
            waveform, 
            (self.window_size // 2, self.window_size // 2),
            mode='constant'
        )
        
        # windows = []
        # for start in range(0, effective_waveform.shape[-1] - self.window_size + 1, self.hop_length):
        #     end = start + self.window_size
        #     windows.append(effective_waveform[start:end])

        # return np.stack(windows)

        starts = np.arange(0, effective_waveform.shape[-1] - self.window_size + 1, self.hop_length)
        offsets = np.arange(self.window_size)
        indics = starts[:, None] + offsets
        return effective_waveform[indics]
    

class Hann:
    def __init__(self, window_size=1024):
        self.sliding_window = scipy.signal.windows.hann(M=window_size, sym=False)
    
    def __call__(self, windows):
        # windows (seq_len, window_length)
        # _sliding_window (window_length,)
        return windows * self.sliding_window[None, :]



class DFT:
    def __init__(self, n_freqs=None):
        self.n_freqs = n_freqs

    def __call__(self, windows):
        # windows (seq_len, window_length)
        fft_windows = np.fft.rfft(windows, n=None, axis=1)
        spec = np.absolute(fft_windows) # (n_frames, window_size // 2) i.e. (time frame, freq coef)
        spec = spec[:, :self.n_freqs]
        return spec


class Square:
    def __call__(self, array):
        return np.square(array)


class Mel:
    def __init__(self, n_fft, n_mels=80, sample_rate=22050):
        self.mel_filterbank = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft, 
            n_mels=n_mels,
            fmin=1,
            fmax=8192,
        )  # (n_mels, 1 + n_fft/2)
        self.inverse_mel_filterbank = np.linalg.pinv(self.mel_filterbank)

    def __call__(self, spec):
        # spec (n_frames, 1 + n_fft/2)
        mel = spec @ self.mel_filterbank.T
        return mel

    def restore(self, mel):
        # mel (n_frames, n_mels)
        spec = mel @ self.inverse_mel_filterbank.T
        return spec


class GriffinLim:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.griffin_lim = partial(
            librosa.griffinlim,
            n_iter=32,
            hop_length=hop_length,
            win_length=window_size,
            n_fft=window_size,
            window='hann'
        )

    def __call__(self, spec):
        return self.griffin_lim(spec.T)


class Wav2Spectrogram:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.windowing = Windowing(window_size=window_size, hop_length=hop_length)
        self.hann = Hann(window_size=window_size)
        self.fft = DFT(n_freqs=n_freqs)
        # self.square = Square()
        self.griffin_lim = GriffinLim(window_size=window_size, hop_length=hop_length, n_freqs=n_freqs)

    def __call__(self, waveform):
        return self.fft(self.hann(self.windowing(waveform)))

    def restore(self, spec):
        return self.griffin_lim(spec)


class Wav2Mel:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None, n_mels=80, sample_rate=22050):
        self.wav_to_spec = Wav2Spectrogram(
            window_size=window_size,
            hop_length=hop_length,
            n_freqs=n_freqs)
        self.spec_to_mel = Mel(
            n_fft=window_size,
            n_mels=n_mels,
            sample_rate=sample_rate)

    def __call__(self, waveform):
        return self.spec_to_mel(self.wav_to_spec(waveform))

    def restore(self, mel):
        return self.wav_to_spec.restore(self.spec_to_mel.restore(mel))


class TimeReverse:
    def __call__(self, mel):
        return mel[::-1, :]


class Loudness:
    def __init__(self, loudness_factor):
        self.loudness_factor = loudness_factor

    def __call__(self, mel):
        return mel * self.loudness_factor



class PitchUp:
    def __init__(self, num_mels_up):
        self.num_mels_up = num_mels_up

    def __call__(self, mel):
        shifted = np.zeros_like(mel)
        shift = max(mel.shape[1] - self.num_mels_up, 0)
        shifted[:, -shift:] = mel[:, :shift]
        return shifted


class PitchDown:
    def __init__(self, num_mels_down):
        self.num_mels_down = num_mels_down

    def __call__(self, mel):
        shifted = np.zeros_like(mel)
        shift = max(mel.shape[1] - self.num_mels_down, 0)
        shifted[:, :shift] = mel[:, -shift:]
        return shifted


class SpeedUpDown:
    def __init__(self, speed_up_factor=1.0):
        self.speed_up_factor = speed_up_factor

    def __call__(self, mel):
        target_n_frames = int(self.speed_up_factor * mel.shape[0])
        new_idx = np.round(np.arange(target_n_frames) / self.speed_up_factor).astype(np.int64)
        new_idx = np.clip(new_idx, 0, mel.shape[0] - 1)
        return mel[new_idx]


class FrequenciesSwap:
    def __call__(self, mel):
        return mel[:, ::-1]


class WeakFrequenciesRemoval:
    def __init__(self, quantile=0.05):
        self.quantile = quantile

    def __call__(self, mel):
        return np.where(mel < np.quantile(mel, q=self.quantile), 0, mel)


class Cringe1:
    def __call__(self, mel):
        return 1 / mel


class Cringe2:
    def __call__(self, mel):
        np.random.seed(123)
        mel = mel.copy()
        n_i = np.random.randint(mel.shape[0])
        i = np.random.choice(mel.shape[0], size=n_i)
        mel[i] = 0
        n_j = np.random.randint(mel.shape[1])
        j = np.random.choice(mel.shape[1], size=n_j)
        mel[:, j] = 0
        return mel