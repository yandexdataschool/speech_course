import random

import lightning as L
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset


SAMPLE_RATE = 16_000
AUDIO_LENGTH = 16_000
LONG_AUDIO_LENGTH = 64_000


def _crop_or_pad(audio: torch.Tensor, length: int, train: bool) -> torch.Tensor:
    n_samples = audio.shape[-1]
    if n_samples <= length:
        return F.pad(audio, (0, length - n_samples))

    start = random.randint(0, n_samples - length) if train else 0
    return audio[..., start:start + length]


class LibriSpeechDataset(Dataset):
    def __init__(self, root: str, url: str = "train-clean-100", train: bool = True):
        self._base = torchaudio.datasets.LIBRISPEECH(root, url=url, download=True)
        self._train = train

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        audio, sample_rate, _, speaker_id, _, _ = self._base[idx]
        if sample_rate != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sample_rate, SAMPLE_RATE)
        return _crop_or_pad(audio, AUDIO_LENGTH, train=self._train), int(speaker_id)

    def get_example(self, idx: int, length: int, train: bool | None = None) -> tuple[torch.Tensor, int]:
        audio, sample_rate, _, speaker_id, _, _ = self._base[idx]
        if sample_rate != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sample_rate, SAMPLE_RATE)
        train = self._train if train is None else train
        return _crop_or_pad(audio, length, train=train), int(speaker_id)


class LibriSpeechDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        train_url: str = "train-clean-100",
        val_url: str = "dev-clean",
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_url = train_url
        self.val_url = val_url
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self) -> None:
        torchaudio.datasets.LIBRISPEECH(self.data_dir, url=self.train_url, download=True)
        torchaudio.datasets.LIBRISPEECH(self.data_dir, url=self.val_url, download=True)

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_ds = LibriSpeechDataset(self.data_dir, self.train_url, train=True)
            self.val_ds = LibriSpeechDataset(self.data_dir, self.val_url, train=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def sample_val_examples(
        self,
        num_examples: int = 4,
        offset: int = 0,
        length: int = LONG_AUDIO_LENGTH,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, "val_ds"):
            self.setup("fit")

        audios = []
        speaker_ids = []
        seen_speakers = set()
        dataset_length = len(self.val_ds)

        for step in range(dataset_length):
            idx = (offset + step) % dataset_length
            audio, speaker_id = self.val_ds.get_example(idx, length=length, train=False)
            if speaker_id in seen_speakers:
                continue
            audios.append(audio)
            speaker_ids.append(speaker_id)
            seen_speakers.add(speaker_id)
            if len(audios) == num_examples:
                break

        if not audios:
            raise RuntimeError("Validation dataset is empty.")

        return torch.stack(audios), torch.tensor(speaker_ids, dtype=torch.long)
