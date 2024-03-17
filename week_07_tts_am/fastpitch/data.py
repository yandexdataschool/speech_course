import json
import os
from dataclasses import dataclass
from functools import partial
from itertools import groupby
from typing import Iterator

import librosa
import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from week_07_tts_am.fastpitch.common.utils import ToDeviceMixin, convert_to_tensor, pad_tensor_list
from week_07_tts_am.fastpitch.sampled_array import SampledArray, resample


def path_to_audio(r, data_dir: str) -> str:
    return os.path.join(data_dir, 'wavs', f'{r.utterance_id}.wav')


def read_utterance(r, data_dir: str) -> str:
    with open(os.path.join(data_dir, 'mfa_aligned', f'{r.utterance_id}.json')) as f:
        utterance = json.load(f)
    return utterance


def get_split(r, eval_share, total_length) -> str:
    if r['index'] < (1 - eval_share) * total_length:
        return 'train'
    return 'eval'


@dataclass
class Wav:
    data: np.array
    sr: int

    @property
    def duration(self):
        return self.data.shape[0] / self.sr


class LocalDataset:
    def __init__(self, data_dir: str, eval_share: float = 0.02, filter_query: str = None):
        self.data_dir = data_dir
        self.eval_share = eval_share
        self.df = self._prepare_df()

        if filter_query:
            self.df = self._apply_filter(filter_query)
   
    def _prepare_df(self):
        df = pd.read_csv(os.path.join(self.data_dir, 'metadata.csv'), sep='|', header=None).dropna()
        df = df.rename(columns={0: 'utterance_id', 1: 'original_text', 2: 'normalized_text'})
        df = df.sort_values(by='utterance_id')
        df = df.reset_index()

        df['audio_path'] = df.apply(partial(path_to_audio, data_dir=self.data_dir), axis=1)
        df['split'] = df.apply(partial(get_split, eval_share=self.eval_share, total_length=len(df)), axis=1)
        df['utterance'] = df.apply(partial(read_utterance, data_dir=self.data_dir), axis=1)
        return df

    def _apply_filter(self, query: str):
        return self.df.query(query)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

    def __len__(self):
        return len(self.df)


class SymbolsSet:
    phonemes = [
        'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2',
        'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
        'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2',
        'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0',
        'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2',
        'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1',
        'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH',
    ]
    special = ['sil', 'spn']
    symbols_to_id = {s: i for i, s in enumerate(phonemes + special)}
    id_to_symbols = {i: s for i, s in enumerate(phonemes + special)}

    def __call__(self, row) -> list[int]:
        phones = row.utterance['tiers']['phones']['entries']
        phonemes = [p for (_, _, p) in phones]
        return [self.symbols_to_id[p] for p in phonemes]

    def encode(self, phonemes: list[str]) -> list[int]:
        return [self.symbols_to_id[p] for p in phonemes]
    
    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_symbols[i] for i in ids]


class MelSpectrogram:
    sample_rate: int = 22050
    min_frequency: int = 0.0
    max_frequency: int = 8000.0
    num_mels: int = 80
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256

    def __init__(self):
        self.mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.num_mels,
            fmin=self.min_frequency,
            fmax=self.max_frequency
        )

    @classmethod
    def time_to_frames(cls, time):
        return librosa.time_to_frames(time, sr=cls.sample_rate, hop_length=cls.hop_length)

    @classmethod
    def frames_to_time(cls, frames):
        return librosa.frames_to_time(frames, sr=cls.sample_rate, hop_length=cls.hop_length)

    def _stft(self, y: np.array) -> np.array:
        return librosa.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hann',
            center=True
        )

    def _linear_to_mel(self, s: np.array) -> np.array:
        return np.dot(self.mel_basis, s)

    def _amp_to_db(self, x: np.array) -> np.array:
        return np.log(np.maximum(1e-5, x))

    def __call__(self, row) -> np.array:
        wav, _ = sf.read(row.audio_path)
        s = np.abs(self._stft(wav))
        m = self._amp_to_db(self._linear_to_mel(s))
        return SampledArray(value=m.T, t1=0, step=self.frames_to_time(1))


class PitchExtractor:
    def __init__(self, normalize_mean: float, normalize_std: float):
        self.mean = np.array(normalize_mean)
        self.std = np.array(normalize_std)

    def _normalize_pitch(self, pitch: np.array) -> np.array:
        zeros = (pitch == 0.0)
        pitch -= self.mean
        pitch /= self.std
        pitch[zeros] = 0.0
        return pitch

    def __call__(self, row, normalize: bool = True) -> np.array:
        sound = parselmouth.Sound(row.audio_path)
        pitch_obj = sound.to_pitch()
        pitch = pitch_obj.selected_array["frequency"]
        pitch[np.isnan(pitch)] = 0
        if normalize:
            pitch = self._normalize_pitch(pitch)
        return SampledArray(value=pitch, t1=pitch_obj.t1, step=pitch_obj.time_step)


class DurationExtractor:
    alignment_step: float = 0.01    # 10 milliseconds

    def _phone_duration(self, start: int, end: int, step: float):
        return round((end - start) / step)

    def __call__(self, row):
        phones = row.utterance['tiers']['phones']['entries']
        durations = []

        for i, (start, end, _) in enumerate(phones):
            durations.append(round((end - start) / self.alignment_step))

        durations = np.array(durations)

        # A hack to avoid cases when short phones disappear after remapping to mel timeline
        for i in range(len(durations)):
            if durations[i] == 1:
                durations[durations.argmax()] -= 1
                durations[i] += 1

        return SampledArray(
            value=np.repeat(np.arange(len(phones)), durations),
            t1=0,
            step=self.alignment_step
        )


def durations_to_intervals(durations: np.array) -> Iterator:
    prev = 0
    for stamp in durations.cumsum():
        yield prev, stamp
        prev = stamp


def average_pitch_over_durations(pitch: SampledArray, durations: np.ndarray) -> np.ndarray:
    values = []
    for start, stop in durations_to_intervals(durations):
        phone_pitch = pitch.value[start:stop]
        values.append(phone_pitch.mean())
    return np.array(values)


class RowFactory:
    def __init__(self):
        self.symbols_set = SymbolsSet()
        self.mel_extractor = MelSpectrogram()
        self.pitch_extractor = PitchExtractor(
            normalize_mean=215.42230,    # LJSpeech constants, in practice depend on the dataset
            normalize_std=62.51305
        )
        self.duration_extractor = DurationExtractor()

    def __call__(self, raw_row):
        symbol_ids = self.symbols_set(raw_row)
        mel_arr = self.mel_extractor(raw_row)
        duration_arr = self.duration_extractor(raw_row)
        pitch_arr = self.pitch_extractor(raw_row, normalize=True)

        pitch_resampled = resample(pitch_arr, mel_arr)
        dur_resampled = resample(duration_arr, mel_arr)

        durations = np.array([len(list(group)) for _, group in groupby(dur_resampled.value)])

        pitch = average_pitch_over_durations(pitch_resampled, durations)

        assert durations.shape[0] == len(symbol_ids), f"Durations and phones don't match: {durations.shape[0] = }, {len(symbol_ids) = } {raw_row.to_dict() = }"
        assert durations.sum() == mel_arr.value.shape[0], f"Durations and spectrogram don't match: {durations.sum() = } {mel_arr.value.shape[0] = }"

        return {
            'symbols_ids': symbol_ids,
            'mel': mel_arr.value.T,
            'pitch': pitch,
            'duration': durations,
            'raw_pitch': pitch_resampled.value,
            'meta': raw_row.to_dict()
        }        


class TTSDataset:
    def __init__(self, data_dir: str, filter_query: str = None):
        self.local_dataset = LocalDataset(data_dir, filter_query=filter_query)
        self.row_factory = RowFactory()

    def __getitem__(self, idx):
        raw_row = self.local_dataset[idx]
        return self.row_factory(raw_row)

    def __len__(self):
        return len(self.local_dataset)


@dataclass
class FastPitchBatch(ToDeviceMixin):
    texts: torch.Tensor
    text_lengths: torch.Tensor
    mels: torch.Tensor = None
    mel_lengths: torch.Tensor = None
    pitches: torch.Tensor = None
    durations: torch.Tensor = None
    paces: torch.Tensor = None


class TextMelCollate:
    def __call__(self, batch: list[dict]) -> FastPitchBatch:
        to_long_type = lambda x: torch.LongTensor(x)
        to_float_type = lambda x: torch.FloatTensor(x)
        
        texts, text_lengths = pad_tensor_list(convert_to_tensor(batch, 'symbols_ids', to_long_type))
        pitches, _ = pad_tensor_list(convert_to_tensor(batch, 'pitch', to_float_type))
        durations, _ = pad_tensor_list(convert_to_tensor(batch, 'duration', to_long_type))

        mels_tensors = convert_to_tensor(batch, 'mel', to_float_type)
        mels, mel_lengths = pad_tensor_list([t.transpose(1, 0) for t in mels_tensors])
        mels = mels.transpose(2, 1)

        return FastPitchBatch(
            texts=texts,
            text_lengths=text_lengths,
            mels=mels,
            mel_lengths=mel_lengths,
            pitches=pitches,
            durations=durations
        )


def get_loader(dataset: LocalDataset, hparams, collate_fn, shuffle: bool = False):
    return DataLoader(
        dataset,
        batch_size=hparams.batch_size,
        num_workers=4,
        shuffle=shuffle,
        collate_fn=collate_fn
    )


def prepare_loaders(data_dir, hparams):
    train_dataset = TTSDataset(data_dir, filter_query="split == 'train'")
    val_dataset = TTSDataset(data_dir, filter_query="split == 'eval'")

    train_loader = get_loader(train_dataset, hparams, TextMelCollate(), shuffle=True)
    val_loader = get_loader(val_dataset, hparams, TextMelCollate(), shuffle=False)
    return train_loader, val_loader
