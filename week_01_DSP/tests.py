import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from matplotlib import pyplot as plt

from transforms import *

test_samples_path = Path(__file__).parent.parent / "data" / "01_DSP" / "tests"
num_test_samples = 10


def test_windowing(testing_class=Windowing, n_repeats=1):
    # Check lengths
    for idx in range(n_repeats):
        wav_size = random.randrange(1000, 5000)
        window_size = random.randrange(16, wav_size // 2)
        hop_length = random.randrange(3, window_size)

        wav = np.random.randn(wav_size)
        transform = testing_class(window_size=window_size, hop_length=hop_length)

        result = transform(wav)

        n_buckets, bucket_size = result.shape
        expected_length = (wav_size - window_size % 2) // hop_length + 1

        if expected_length != n_buckets or window_size != bucket_size:
            msg = f"We were expecting result of size [(wav_size - window_size % 2) // hop_length + 1, window_size].\n" \
                  f"But for the parameters {wav_size=} {window_size=} {hop_length=}\n" \
                  f"We were expecting {(expected_length, window_size)}, got {result.shape=}"
            print(msg)
            return False

    # Check zero padding even
    wav = np.ones(128, dtype=int)
    transform = testing_class(window_size=64, hop_length=16)
    
    result = transform(wav)

    sum_pad_values_0 = np.absolute(result[0, :32]).sum()
    sum_pad_values_1 = np.absolute(result[1, :16]).sum()
    sum_pad_values_2 = np.absolute(result[-2, -16:]).sum()
    sum_pad_values_3 = np.absolute(result[-1, -32:]).sum()
    sum_pad_values_start = sum_pad_values_0 + sum_pad_values_1
    sum_pad_values_end = sum_pad_values_2 + sum_pad_values_3

    if sum_pad_values_start > 0:
        print("Not enough padding values in the start")
        return False
    if sum_pad_values_end > 0:
        print("Not enough padding values in the end")
        return False
    
    # Check zero padding odd
    wav = np.ones(128, dtype=int)
    transform = testing_class(window_size=63, hop_length=16)
    
    result = transform(wav)

    sum_pad_values_0 = np.absolute(result[0, :31]).sum()
    sum_pad_values_1 = np.absolute(result[1, :15]).sum()
    sum_pad_values_2 = np.absolute(result[-1, -16:]).sum()
    sum_pad_values_start = sum_pad_values_0 + sum_pad_values_1
    sum_pad_values_end = sum_pad_values_2

    if sum_pad_values_start > 0:
        print("Not enough padding values in the start")
        return False
    if sum_pad_values_end > 0:
        print("Not enough padding values in the end")
        return False

    # Test on random noize samples
    for idx in range(num_test_samples):
        pickle_path = test_samples_path / "windowing" / f"{idx:02}.pkl"
        wav, trg, wav_size, window_size, hop_length = pickle.load(pickle_path.open("rb"))
        transform = testing_class(window_size=window_size, hop_length=hop_length)
        result = transform(wav)

        if not np.allclose(result, trg):
            msg = f"Didn't work for noise sample number {idx}.\n" \
                  f"Input of shape {wav.shape} expected result of shape {trg.shape}.\n" \
                  f"Got result of shape {result.shape}, but the target and result do not match.\n"
            print(msg)
            return False

    return True


def test_hann(testing_class=Hann):
    for idx in range(num_test_samples):
        pickle_path = test_samples_path / "hann" / f"{idx:02}.pkl"
        inp, trg, window_size, n_frames = pickle.load(pickle_path.open("rb"))

        transform = testing_class(window_size=window_size)
        result = transform(inp)

        if not np.allclose(result, trg):
            msg = f"Didn't work for noise sample number {idx}.\n" \
                  f"Input of shape {inp.shape} expected result of shape {trg.shape}.\n" \
                  f"Got result of shape {result.shape}, but the target and result do not match.\n"
            print(msg)
            return False

    return True


def test_dft(testing_class=DFT):
    for idx in range(num_test_samples):
        pickle_path = test_samples_path / "dft" / f"{idx:02}.pkl"
        inp, trg, window_size, n_frames, n_freqs = pickle.load(pickle_path.open("rb"))
        
        trans = testing_class(n_freqs=n_freqs)
        result = trans(inp)

        if result.shape != trg.shape:
            print(f"Shapes do not match {trg.shape=} != {result.shape=}")
            return False
        if not np.allclose(result, trg):
            msg = f"Didn't work for noise sample number {idx}.\n" \
                  f"Input of shape {inp.shape} expected result of shape {trg.shape}.\n" \
                  f"Got result of shape {result.shape}, but the target and result do not match.\n"
            print(msg)
            return False
    return True


def test_mel(testing_class=Mel):
    for idx in range(num_test_samples):
        pickle_path = test_samples_path / "mel" / f"{idx:02}.pkl"
        inp, trg, _, n_fft, spec_size, n_frames, n_mels = pickle.load(pickle_path.open("rb"))
        
        trans = testing_class(n_fft=n_fft, n_mels=n_mels, sample_rate=22050)
        result = trans(inp)

        if result.shape != trg.shape:
            print(f"Shapes do not match {trg.shape=} != {result.shape=}")
            return False
        if not np.allclose(result, trg):
            msg = f"Didn't work mel forward for noisy sample number {idx}.\n" \
                  f"Input of shape {inp.shape} expected result of shape {trg.shape}.\n" \
                  f"Got result of shape {result.shape}, but the target and result do not match.\n"
            print(msg)
            return False

    for idx in range(num_test_samples):
        pickle_path = test_samples_path / "mel" / f"{idx:02}.pkl"
        _, inp, trg, n_fft, spec_size, n_frames, n_mels = pickle.load(pickle_path.open("rb"))
        
        trans = testing_class(n_fft=n_fft, n_mels=n_mels, sample_rate=22050)
        result = trans.restore(inp)

        if result.shape != trg.shape:
            print(f"")
            print(f"Shapes do not match {trg.shape=} != {result.shape=}")
            return False
        if not np.allclose(result, trg):
            msg = f"Didn't manage to restore noisy sample number {idx}.\n" \
                  f"Input of shape {inp.shape} expected result of shape {trg.shape}.\n" \
                  f"Got result of shape {result.shape}, but the target and result do not match.\n"
            print(msg)
            return False

    return True


# Doesn't work as torch.stft does not match np stft
def test_spectrogram(n_repeats=100):
    for test_idx in range(n_repeats):
        waveform_len = np.random.randint(128, 4096)
        waveform = np.random.randn(waveform_len)

        n_fft = np.random.randint(16, waveform_len // 4)
        window_size = np.random.randint(4, n_fft - 1)
        hop_length = np.random.randint(1, window_size - 1)
        print(f"{waveform_len=} {n_fft=} {window_size=} {hop_length=}")

        np_wav_2_spec = Wav2Spectrogram(
            window_size=window_size,
            hop_length=hop_length,
            n_fft=n_fft)
        
        torch_wav_2_spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=window_size,
            hop_length=hop_length,
            center=True,
            pad_mode="constant",
            power=1)

        np_spec = np_wav_2_spec(waveform)
        # print(f"{torch.tensor(waveform).shape=}")
        torch_spec = torch_wav_2_spec(torch.tensor(waveform)).numpy().T

        if np_spec.shape != torch_spec.shape:
            print(f"[{test_idx}] sizes mismatch")
            print(f"{window_size=} {hop_length=} {n_fft=}")
            print(f"torch: {torch_spec.shape} numpy: {np_spec.shape}")
            raise AssertionError()
        
        if not np.allclose(torch_spec, np_spec, rtol=0.3, atol=0.3):
            diff = (np_spec - torch_spec)
            print(f"Mean square error: {np.sqrt((diff ** 2).mean())}")
            print(f"Max diff: {np.abs(diff).max()}")
            plt.imshow(diff)
            plt.colorbar()
            plt.show()
            print(f"relative_error : {np.absolute(diff / torch_spec).max()}")
            plt.imshow(diff / torch_spec)
            plt.colorbar()
            plt.show()
            plt.imshow(torch_spec)
            plt.colorbar()
            plt.show()
            plt.imshow(np_spec)
            plt.colorbar()
            plt.show()
            raise AssertionError()
    return True
