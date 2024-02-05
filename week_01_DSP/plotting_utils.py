import librosa
import numpy as np
import scipy
from matplotlib import pyplot as plt
from IPython.display import display, Audio
from cmath import pi, e, polar

from transforms import *

# TODO: find a nice colormap
# Lovely ones: cividis gist_earth_r inferno_r nipy_spectral_r
spectrogram_colormap = plt.cm.nipy_spectral_r


def plot_wav(wav, sample_rate=22050, end=None, ax=None):
    show_needed = ax is None
    if show_needed:
        fig, ax = plt.subplots(figsize=(15, 5))

    if end is not None:
        wav=wav[:end]
    ax.set_title(f"{len(wav) / sample_rate:.2f} seconds of sound. {len(wav)} amplitudes.")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.plot(wav[:end])
    for sec in range(0, len(wav), sample_rate):
        ax.axvline(sec, color="grey")
    if show_needed:
        fig.show()


def plot_windowed_wav(windows):
    fig, ax = plt.subplots(figsize=(10, 5))

    max_abs = max(np.abs(np.min(windows)), np.abs(np.max(windows)))
    im = ax.imshow(windows.T, aspect='auto', cmap=plt.cm.bwr, vmin=-max_abs, vmax=max_abs)
    plt.colorbar(im, ax=ax, use_gridspec=True)

    ax.set_xlabel("Window index")
    ax.set_ylabel("Samples")

    fig.show()

def plot_dft(spec):
    fig, ax = plt.subplots(figsize=(10, 5))

    im = ax.imshow(spec.T, aspect='auto', cmap=spectrogram_colormap, origin="lower")
    plt.colorbar(im, ax=ax, use_gridspec=True)

    ax.set_xlabel("Window index")
    ax.set_ylabel("Frequencies")

    fig.show()


def plot_hann_window(window_size=1024, ax=None):
    show_needed = ax is None
    if show_needed:
        fig, ax = plt.subplots(figsize=(15, 5))

    hann_weights = scipy.signal.windows.hann(window_size, sym=False)

    ax.plot(hann_weights)
    ax.grid()
    ax.set_title(f"Hann window")
    ax.set_xlabel("Weights' indices")
    ax.set_ylabel("Weights")
    if show_needed:
        fig.show()


def plot_wav_with_offset(idx, wav, offset, total_length, ax=None):
    assert ax is not None

    ax.plot(np.arange(offset, offset + wav.shape[0]), wav)
    ax.axhline(0, 0, total_length, c="grey", linestyle="dashed")

    ax.set_title(f"Window No{idx}")
    ax.set_ylabel("Amplitude")


def plot_windowing(waveform, windowing_class=Windowing, n_subpictures=4):
    fig, axes = plt.subplots(
        nrows=n_subpictures + 1,
        ncols=1,
        figsize=(15, 4 * (n_subpictures + 1)))

    n_hops_in_win = 4
    hop_length = waveform.shape[0] // (n_subpictures + n_hops_in_win - 1)
    window_size = hop_length * n_hops_in_win
    print(f"{window_size=} {hop_length=} {n_hops_in_win=}")

    windower = windowing_class(window_size=window_size, hop_length=hop_length)
    windows = windower(waveform)

    # Removing paddings
    pad_size = n_hops_in_win // 2
    windows = windows[pad_size:-pad_size, :]

    plot_wav(waveform, ax=axes[0])
    for idx, (window, ax) in enumerate(zip(windows, axes[1:])):
        plot_wav_with_offset(
            idx=idx,
            wav=window,
            offset=hop_length * idx,
            total_length=waveform.shape[0],
            ax=ax)

    x_pad, y_coef = 100, 1.1
    xlim = (- x_pad, len(waveform) + x_pad)
    ylim = (y_coef * waveform.min(), y_coef * waveform.max())
    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid()

    fig.show()


def get_demo_tensor(waveforms):
    window_size = 2048
    for waveform in waveforms:
        assert waveform.shape[0] == window_size

    demo_wavs = [
        np.ones(window_size),
        np.sin(np.linspace(0, 4 * np.pi, window_size)),
        np.sin(np.linspace(0, 3 * np.pi, window_size)),
        np.sin(np.linspace(0.5 * np.pi, 3.5 * np.pi, window_size)),
        # np.tanh(np.linspace(-10, 10, window_size)),
        np.sin(np.linspace(-np.pi / 2, np.pi / 2, window_size)),
        *waveforms,
    ]
    return np.vstack(demo_wavs)


def plot_hann(demo_tensor, hann_class=Hann):
    col_names = ["Raw wavs", "Hann wav", "Raw fft amplitudes", "Hann fft amplitdes"]
    line_names = ["Constant", "Sine 4pi", "Sine 3pi", "Sine 3pi + phi", "Slow sine", "wav_0", "wav_1"]

    fig, axes = plt.subplots(
        nrows=demo_tensor.shape[0],
        ncols=4,
        figsize=(15, 3 * (demo_tensor.shape[0])),
        squeeze=False)

    n_fft = 64
    hann = hann_class(window_size=demo_tensor.shape[1])

    raw_wavs = demo_tensor
    hann_wavs = hann(raw_wavs)
    raw_amps = np.absolute(np.fft.rfft(raw_wavs))[:, :n_fft]
    hann_amps = np.absolute(np.fft.rfft(hann_wavs))[:, :n_fft]

    for idx, line in enumerate(axes):
        raw_wav, hann_wav = raw_wavs[idx], hann_wavs[idx]
        raw_amp, hann_amp = raw_amps[idx], hann_amps[idx]
        wav_ylim = (1.1 * min(raw_wav.min(), hann_wav.min()), 1.1 * max(raw_wav.max(), hann_wav.max()))
        amp_ylim = (1.1 * min(raw_amp.min(), hann_amp.min()), 1.1 * max(raw_amp.max(), hann_amp.max()))

        line[0].plot(raw_wav)
        line[0].axhline(0, 0, len(raw_wav), color="grey", linestyle="dashed")
        line[0].set_ylim(*wav_ylim)

        line[1].plot(hann_wav)
        line[1].axhline(0, 0, len(hann_wav), color="grey", linestyle="dashed")
        line[1].set_ylim(*wav_ylim)

        line[2].vlines(
            x=np.arange(raw_amp.shape[0]),
            ymin=0, ymax=raw_amp)
        line[2].set_ylim(*amp_ylim)

        line[3].vlines(
            x=np.arange(hann_amp.shape[0]),
            ymin=0, ymax=hann_amp)
        line[3].set_ylim(*amp_ylim)

        # Setting names
        if idx == 0:
            for ax, name in zip(line, col_names):
                ax.set_title(name)
        line[0].set_ylabel(line_names[idx])

    fig.show()


def plot_signal_fft(signal_fn, sample_rate, window_size):
    x_ticks = np.arange(sample_rate)
    signal = signal_fn(x_ticks / sample_rate)

    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(15, 4))

    ax0.plot(
        np.linspace(0, sample_rate, 2048),
        signal_fn(np.linspace(0, 1, 2048)),
        label="Continuous signal")
    ax0.scatter(x_ticks, signal, c="black", s=5, label="Sampled signal")
    ax0.axvline(0, c="grey", ls="--")
    ax0.axvline(window_size, c="grey", ls="--", label="Window_size")
    ax0.axhline(np.mean(signal), c="black", linewidth=0.5, label="Mean")

    ax0.set_xlabel("Indices")
    ax0.set_ylabel("Signal values")
    ax0.legend()

    window = signal[:window_size]

    # ax1.plot(np.arange(window.shape[0]), window)
    ax1.plot(
        np.linspace(0, window_size, 2048),
        signal_fn(np.linspace(0, window_size / sample_rate, 2048)),
        label="Continuous signal")
    ax1.scatter(np.arange(window.shape[0]), window, c='black', s=20, label="Sampled signal")
    ax1.axhline(np.mean(window), c="black", linewidth=0.5, label="Mean")

    ax1.set_xlabel("Indices")
    ax1.set_ylabel("Signal values")
    ax1.legend()

    fft = np.fft.rfft(window)

    ax2.vlines(
        x=np.arange(fft.shape[0]) * sample_rate / window_size,
        ymin=0,
        ymax=np.absolute(fft),
        color="blue")
    ax2.axhline(0, linewidth=0.5, c='black')

    ax2.set_xlabel("FFT frequencies 1/s")
    ax2.set_ylabel("Magnitudes of FFT values |FFT|")

    n_labels = 6
    labels = []
    for idx in (np.arange(fft.shape[0]) * sample_rate / window_size):
        if idx % int(fft.shape[0] // n_labels) == 0 or idx == (np.arange(fft.shape[0]) * sample_rate / window_size)[-1]:
            labels.append(str(int(idx)))
        else:
            labels.append("")

    ax2.set_xticks(
        ticks=np.arange(fft.shape[0]) * sample_rate / window_size,
        labels=labels,
    )

    fig.show()


def plot_mel_scale():
    mel_scale = librosa.filters.mel(sr=22050, n_fft=512, n_mels=80, fmin=1, fmax=8192)
    fig, ax = plt.subplots(figsize=(15, 5))
    im = ax.imshow(mel_scale, origin="lower", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, use_gridspec=True)

    ax.set_xlabel("FFT frequencies")
    ax.set_ylabel("Mel frequencies")
    ax.set_title("Mel weights")

    fig.show()


def plot_fft_difference():
    arrays = [
        np.hstack((np.zeros(16), np.ones(32), np.zeros(16))),
        np.sin(np.linspace(- np.pi, np.pi, 64)),
        np.random.randn(64),
    ]
    col_names = [
        "Waveform",
        "Magnitude of FFT transform",
        "Phase of FFT transform",
        "Magnitude value of RFFT transform",
        "Phase of RFFT transform",
    ]
    fig, axes = plt.subplots(nrows=len(arrays), ncols=5, figsize=(15, 6), squeeze=False)

    for idx, (line, arr) in enumerate(zip(axes, arrays)):
        fft_spec = np.fft.fft(arr)
        fft_freqs = np.fft.fftfreq(len(arr))
        # indices = np.argsort(fft_freqs)
        # fft_freqs, fft_spec = fft_freqs[indices], fft_spec[indices]

        rfft_spec = np.fft.rfft(arr)
        rfft_freqs = np.fft.rfftfreq(len(arr))
        # indices = np.argsort(rfft_freqs)
        # rfft_freqs, rfft_spec = rfft_freqs[indices], rfft_spec[indices]

        line[0].plot(np.arange(len(arr)), arr)
        line[0].set_xlabel("Samples")
        line[0].set_ylabel("Amplitude")

        line[1].vlines(
            x=fft_freqs,
            ymin=0, ymax=np.absolute(fft_spec))
        line[1].axhline(0, color="grey", linestyle="dashed")
        line[1].set_xlabel("Frequencies")

        line[2].vlines(
            x=fft_freqs,
            ymin=0, ymax=np.angle(fft_spec))
        line[2].axhline(0, color="grey", linestyle="dashed")
        line[2].set_xlabel("Frequencies")

        line[3].vlines(
            x=rfft_freqs,
            ymin=0, ymax=np.absolute(rfft_spec))
        line[3].axhline(0, color="grey", linestyle="dashed")
        line[3].set_xlabel("Frequencies")

        line[4].vlines(
            x=rfft_freqs,
            ymin=0, ymax=np.angle(rfft_spec))
        line[4].axhline(0, color="grey", linestyle="dashed")
        line[4].set_xlabel("Frequencies")

        if idx == 0:
            for ax, name in zip(line, col_names):
                ax.set_title(name)

    fig.tight_layout()
    fig.show()


def plot_spec(spec, ax=None, spec_type="spec", title=None, colorbar=True, **kwargs):
    show_needed = ax is None
    if show_needed:
        fig, ax = plt.subplots(figsize=(15, 5))

    im_ratio = spec.shape[1] / spec.shape[0]
    im = ax.imshow(spec.T, origin="lower", cmap=spectrogram_colormap, aspect=1, **kwargs)

    if colorbar:
        plt.colorbar(im, ax=ax, use_gridspec=True, fraction=0.047 * im_ratio)

    ax.set_xlabel("Time (frames)")
    spec_to_label = {"spec": "Frequencies", "mel": "Mel indices", "windows": "Samples"}
    ax.set_ylabel(spec_to_label.get(spec_type, ""))
    if title is not None:
        ax.set_title(title)

    if show_needed:
        fig.show()


def plot_spec_mel(spec, mel):
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(8, 8), sharex=True, height_ratios=[512, 80])
    # fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(15, 5), width_ratios=[1, 1])

    # TODO: Retrun colorbar
    plot_spec(spec, ax=ax0, spec_type="spec", title="Spectrogram", colorbar=False)
    plot_spec(mel, ax=ax1, spec_type="mel", title="Mel", colorbar=False)

    fig.tight_layout()
    fig.show()


def plot_transformed_mels(mel, transformed_mel):
    fig, (ax0, ax1) = plt.subplots(
        ncols=2,
        figsize=(15, 5),
        gridspec_kw={"width_ratios": [mel.shape[0], transformed_mel.shape[0]]})

    max_value = max(mel.max(), transformed_mel.max())

    plot_spec(mel, ax=ax0, spec_type="mel", title="Mel", vmin=0., vmax=max_value)
    plot_spec(transformed_mel, ax=ax1, spec_type="mel", title="Mel spectrogram", vmin=0., vmax=max_value)

    # fig.tight_layout()
    fig.show()


def plot_rerstored_spec(spec, restored_spec):
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1, 1]})

    max_value = max(spec.max(), restored_spec.max())

    plot_spec(spec, ax=ax0, spec_type="spec", title="Spectrogram", vmin=0., vmax=max_value)
    plot_spec(restored_spec, ax=ax1, spec_type="spec", title="Restored spectrogram", vmin=0., vmax=max_value)
    plot_spec(np.abs(spec - restored_spec), ax=ax2, spec_type="spec", title="Difference", vmin=0., vmax=max_value)

    fig.tight_layout()
    fig.show()


def plot_wav_and_mel(wav_dict, sample_rate=22050, wav2mel=None):
    if wav2mel is None:
        wav2mel = Wav2Mel(
            window_size=1024,
            hop_length=256,
            n_freqs=None,
            n_mels=80,
            sample_rate=sample_rate)
    
    keys = sorted(list(wav_dict.keys()))
    colorbar_max = 2.0

    for key in keys:
        print(key)
        wav = wav_dict[key]

        display(Audio(wav, rate=sample_rate))
        fig, (ax0, ax1) = plt.subplots(ncols=2, nrows=1, figsize=(20, 4), gridspec_kw={"width_ratios": [1, 1.5]})
        plot_wav(wav, sample_rate=sample_rate, end=None, ax=ax0)

        mel = wav2mel(wav)
        plot_spec(
            np.clip(mel, a_min=0., a_max=colorbar_max),
            ax=ax1,
            spec_type="mel",
            title=key,
            vmin=0,
            vmax=colorbar_max)
        plt.show()
        fig.tight_layout()
