from typing import Optional
from dataclasses import dataclass


@dataclass
class HParamsFastpitch:
    num_steps: int = 3000
    eval_interval: int = 500
    max_val_audios: int = 20    # Number of samples to log mels, audios, etc.

    batch_size: int = 10
    symbols_embedding_dim: int = 384

    n_mel_channels: int = 80

    # input fft params
    in_fft_n_layers: int = 6
    in_fft_n_heads: int = 1
    in_fft_d_head: int = 64
    in_fft_conv1d_kernel_size: int = 3
    p_in_fft_dropout: float = 0.1
    p_in_fft_dropatt: float = 0.1
    p_in_fft_dropemb: float = 0.0
    encoder_embedding_dim: int = 384

    # output fft params
    out_fft_n_layers: int = 6
    out_fft_n_heads: int = 1
    out_fft_d_head: int = 64
    out_fft_conv1d_kernel_size: int = 3
    p_out_fft_dropout: float = 0.1
    p_out_fft_dropatt: float = 0.1
    p_out_fft_dropemb: float = 0.0

    # duration predictor parameters
    dur_predictor_kernel_size: int = 3
    dur_predictor_filter_size: int = 256
    p_dur_predictor_dropout: float = 0.1
    dur_predictor_n_layers: int = 2

    # pitch predictor parameters
    pitch_predictor_kernel_size: int = 3
    pitch_predictor_filter_size: int = 256
    p_pitch_predictor_dropout: float = 0.1
    pitch_predictor_n_layers: int = 2

    # loss function parameters
    dur_predictor_loss_scale: float = 0.1
    pitch_predictor_loss_scale: float = 0.1

    # optimization parameters
    optimizer: str = 'adam'
    learning_rate: float = 0.01
    weight_decay: float = 1e-6
    grad_clip_thresh: float = 1000.0
    warmup_steps: int = 1000

    # other
    seed: int = 1234

    @classmethod
    def create(cls, dictionary: Optional[dict] = None):
        hparams = cls()
        for key, value in dictionary.items():
            if not hasattr(cls, key):
                raise RuntimeError(f'Unknown attribute: {key}')
            setattr(hparams, key, value)
        return hparams