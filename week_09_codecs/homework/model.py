from itertools import chain
from typing import Optional

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import remove_parametrizations
from torch.nn.utils.parametrizations import weight_norm

from quantizers import Perplexity


def weight_norm_conv1d(*args, **kwargs) -> nn.Conv1d:
    return weight_norm(nn.Conv1d(*args, **kwargs))


def weight_norm_conv_transpose1d(*args, **kwargs) -> nn.ConvTranspose1d:
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def remove_weight_norm(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)) and hasattr(module, "parametrizations"):
        if "weight" in module.parametrizations:
            remove_parametrizations(module, "weight", leave_parametrized=True)


class MultiResolutionSTFTLoss(nn.Module):
    SCALES = [
        (512, 128),
        (1024, 256),
        (2048, 512),
    ]

    def __init__(self, sample_rate: int = 16_000):
        super().__init__()
        for n_fft, _ in self.SCALES:
            self.register_buffer(f"window_{n_fft}", torch.hann_window(n_fft))

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        x_hat = x_hat.squeeze(1)
        total = torch.tensor(0.0, device=x.device)
        for n_fft, hop_length in self.SCALES:
            window = getattr(self, f"window_{n_fft}")
            orig = torch.stft(
                x,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True,
            ).abs().clamp(min=1e-5)
            recon = torch.stft(
                x_hat,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True,
            ).abs().clamp(min=1e-5)

            spectral_convergence = torch.linalg.vector_norm(recon - orig, dim=(-2, -1)) / (
                torch.linalg.vector_norm(orig, dim=(-2, -1)) + 1e-5
            )
            log_magnitude = F.l1_loss(torch.log(recon), torch.log(orig))
            total = total + spectral_convergence.mean() + log_magnitude
        return total / len(self.SCALES)


class ResidualBlock1d(nn.Module):
    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ELU(inplace=True),
            weight_norm_conv1d(channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ELU(inplace=True),
            weight_norm_conv1d(hidden_channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualStack1d(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(
            [ResidualBlock1d(channels, hidden_channels) for _ in range(n_layers)]
        )
        self.out_activation = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.out_activation(x)


class AudioEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 64,
        embedding_dim: int = 128,
        n_residual_layers: int = 2,
    ):
        super().__init__()
        C = channels
        self.net = nn.Sequential(
            weight_norm_conv1d(in_channels, C, kernel_size=7, padding=3),
            nn.ELU(inplace=True),
            weight_norm_conv1d(C, C * 2, kernel_size=4, stride=2, padding=1),
            ResidualStack1d(C * 2, C, n_residual_layers),
            weight_norm_conv1d(C * 2, C * 4, kernel_size=4, stride=2, padding=1),
            ResidualStack1d(C * 4, C * 2, n_residual_layers),
            weight_norm_conv1d(C * 4, C * 4, kernel_size=4, stride=2, padding=1),
            ResidualStack1d(C * 4, C * 2, n_residual_layers),
            weight_norm_conv1d(C * 4, embedding_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AudioDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 1,
        channels: int = 64,
        embedding_dim: int = 128,
        n_residual_layers: int = 2,
    ):
        super().__init__()
        C = channels
        self.net = nn.Sequential(
            weight_norm_conv1d(embedding_dim, C * 4, kernel_size=3, padding=1),
            ResidualStack1d(C * 4, C * 2, n_residual_layers),
            weight_norm_conv_transpose1d(C * 4, C * 4, kernel_size=4, stride=2, padding=1),
            ResidualStack1d(C * 4, C * 2, n_residual_layers),
            weight_norm_conv_transpose1d(C * 4, C * 2, kernel_size=4, stride=2, padding=1),
            ResidualStack1d(C * 2, C, n_residual_layers),
            weight_norm_conv_transpose1d(C * 2, C, kernel_size=4, stride=2, padding=1),
            nn.ELU(inplace=True),
            weight_norm_conv1d(C, out_channels, kernel_size=7, padding=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AudioAutoEncoder(L.LightningModule):
    def __init__(
        self,
        quantizer: Optional[nn.Module] = None,
        vq_loss: Optional[nn.Module] = None,
        spectral_weight: float = 0.0,
        waveform_weight: float = 1.0,
        channels: int = 64,
        embedding_dim: int = 128,
        n_residual_layers: int = 2,
        lr: float = 3e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["quantizer", "vq_loss"])

        self.encoder = AudioEncoder(1, channels, embedding_dim, n_residual_layers)
        self.decoder = AudioDecoder(1, channels, embedding_dim, n_residual_layers)
        self.quantizer = quantizer
        self.vq_loss_fn = vq_loss
        self.spectral_loss_fn = MultiResolutionSTFTLoss() if spectral_weight > 0 else None
        self.spectral_weight = spectral_weight
        self.waveform_weight = waveform_weight

        if quantizer is not None and hasattr(quantizer, "codebook_size"):
            self.perplexity_fn = Perplexity(quantizer.codebook_size)
        else:
            self.perplexity_fn = None
        self._weight_norm_removed = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z_q = self.quantizer(z) if self.quantizer is not None else z
        return self.decoder(z_q)

    def prepare_for_inference(self):
        if not self._weight_norm_removed:
            self.encoder.apply(remove_weight_norm)
            self.decoder.apply(remove_weight_norm)
            self._weight_norm_removed = True
        return self.eval()

    def _recon_loss(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=x.device)
        if self.spectral_loss_fn is not None:
            spec = self.spectral_loss_fn(x_hat, x)
            self.log("train/spec_loss", spec)
            loss = loss + self.spectral_weight * spec
        if self.waveform_weight > 0:
            waveform = F.l1_loss(x_hat, x)
            self.log("train/l1_loss", waveform)
            loss = loss + self.waveform_weight * waveform
        return loss

    def _compute_vq_loss(self, z: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor | float:
        if self.vq_loss_fn is None:
            return 0.0
        return self.vq_loss_fn(z, z_q)

    def training_step(self, batch, batch_idx):
        waveform, _ = batch
        if self.quantizer is None:
            loss = self._recon_loss(self.decoder(self.encoder(waveform)), waveform)
        else:
            z = self.encoder(waveform)
            if hasattr(self.quantizer, "quantize_with_loss"):
                z_q, _, vq_loss = self.quantizer.quantize_with_loss(z, self.vq_loss_fn)
            else:
                z_q = self.quantizer(z)
                vq_loss = self._compute_vq_loss(z, z_q)
            if isinstance(vq_loss, torch.Tensor):
                self.log("train/vq_loss", vq_loss)
            # Quantizers that implement their own STE (e.g. FSQ) set
            # uses_straight_through_estimator = False. For those, z_q already
            # carries gradients; applying an outer STE would block them.
            if getattr(self.quantizer, "uses_straight_through_estimator", True):
                z_q_st = z + (z_q - z).detach()
            else:
                z_q_st = z_q
            loss = self._recon_loss(self.decoder(z_q_st), waveform) + vq_loss
        self.log("train/loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        waveform, _ = batch
        z = self.encoder(waveform)

        if self.quantizer is not None:
            if hasattr(self.quantizer, "quantize_with_loss"):
                z_q, indices, vq_loss = self.quantizer.quantize_with_loss(z, self.vq_loss_fn)
            else:
                indices = self.quantizer.encode(z)
                z_q = self.quantizer.decode(indices)
                vq_loss = self._compute_vq_loss(z, z_q)
        else:
            indices = None
            z_q = z
            vq_loss = torch.tensor(0.0, device=z.device)

        x_hat = self.decoder(z_q)
        spec_loss = self.spectral_loss_fn(x_hat, waveform) if self.spectral_loss_fn is not None else torch.tensor(0.0, device=z.device)
        waveform_loss = F.l1_loss(x_hat, waveform)
        loss = self.spectral_weight * spec_loss + self.waveform_weight * waveform_loss + vq_loss

        self.log("val/spec_loss", spec_loss, on_epoch=True)
        self.log("val/l1_loss", waveform_loss, on_epoch=True)
        self.log("val/vq_loss", vq_loss, on_epoch=True)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        if self.perplexity_fn is not None and indices is not None:
            usage_indices = self._usage_indices(indices)
            if usage_indices.dim() == 2:
                usage_indices = usage_indices.unsqueeze(1)

            perps = []
            for k, layer_idx in enumerate(usage_indices.unbind(dim=1), start=1):
                p = self.perplexity_fn(layer_idx.reshape(-1))
                self.log(f"val/perplexity_cb{k}", p, on_epoch=True, prog_bar=(k == 1))
                perps.append(p)

            if len(perps) > 1:
                self.log("val/perplexity_mean", torch.stack(perps).mean(), on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            self._log_validation_audio_examples(waveform)
        return loss

    @torch.no_grad()
    def _sample_logging_examples(self, fallback_audios: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return fallback_audios[:2], None

        audios, example_ids = datamodule.sample_val_examples(
            num_examples=min(2, fallback_audios.shape[0]),
            offset=42,
        )
        return audios.to(self.device), example_ids

    @torch.no_grad()
    def _log_validation_audio_examples(self, fallback_audios: torch.Tensor) -> None:
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return

        audios, example_ids = self._sample_logging_examples(fallback_audios)
        z = self.encoder(audios)
        if self.quantizer is not None:
            indices = self.quantizer.encode(z)
            z_q = self.quantizer.decode(indices)
        else:
            z_q = z
        reconstructions = self.decoder(z_q)

        for i in range(audios.shape[0]):
            suffix = f"id_{int(example_ids[i])}" if example_ids is not None else f"clip_{i}"
            self.logger.experiment.add_audio(
                f"audio/original_{suffix}",
                audios[i].detach().cpu(),
                global_step=self.current_epoch,
                sample_rate=16_000,
            )
            self.logger.experiment.add_audio(
                f"audio/reconstructed_{suffix}",
                reconstructions[i].detach().cpu(),
                global_step=self.current_epoch,
                sample_rate=16_000,
            )

    def _usage_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if self.quantizer is not None and hasattr(self.quantizer, "codes_to_indices"):
            return self.quantizer.codes_to_indices(indices)
        return indices

    @torch.no_grad()
    def compute_codebook_usage(
        self,
        dataloader,
        max_batches: int = 20,
    ) -> dict[str, torch.Tensor] | None:
        if self.quantizer is None or not hasattr(self.quantizer, "encode"):
            return None

        was_training = self.training
        self.eval()
        all_indices = []
        for i, (waveform, _) in enumerate(dataloader):
            if i >= max_batches:
                break
            z = self.encoder(waveform.to(self.device))
            indices = self._usage_indices(self.quantizer.encode(z))
            if indices.dim() == 2:
                indices = indices.unsqueeze(1)
            all_indices.append(indices.cpu())

        if not all_indices:
            if was_training:
                self.train()
            raise RuntimeError("No batches were processed when computing codebook usage.")

        stacked_indices = torch.cat(all_indices, dim=0)
        counts = []
        probabilities = []
        perplexities = []
        for layer_indices in stacked_indices.unbind(dim=1):
            flat_indices = layer_indices.reshape(-1)
            layer_counts = torch.bincount(flat_indices, minlength=self.quantizer.codebook_size)
            layer_probs = layer_counts.float() / layer_counts.sum().clamp(min=1)
            counts.append(layer_counts)
            probabilities.append(layer_probs)
            perplexities.append(self.perplexity_fn(flat_indices) if self.perplexity_fn is not None else torch.tensor(float("nan")))

        if was_training:
            self.train()

        return {
            "counts": torch.stack(counts),
            "probabilities": torch.stack(probabilities),
            "perplexity": torch.stack(perplexities).cpu(),
        }

    @torch.no_grad()
    def plot_codebook_usage(
        self,
        usage: dict[str, torch.Tensor],
    ):
        counts = usage["counts"]
        probabilities = usage["probabilities"]
        perplexities = usage["perplexity"]

        fig, axes = plt.subplots(
            counts.shape[0],
            1,
            figsize=(12, 3 * counts.shape[0]),
            squeeze=False,
            sharex=True,
        )
        entries = torch.arange(counts.shape[1]).numpy()

        for layer_idx, ax in enumerate(axes[:, 0]):
            ax.bar(entries, probabilities[layer_idx].numpy(), width=0.9)
            ax.set_ylabel("fraction")
            title = "Codebook usage" if counts.shape[0] == 1 else f"Codebook {layer_idx + 1}"
            ax.set_title(f"{title}  |  perplexity = {perplexities[layer_idx].item():.1f}")
            ax.set_xlim(-0.5, counts.shape[1] - 0.5)

        axes[-1, 0].set_xlabel("entry")
        fig.tight_layout()
        return fig

    def configure_optimizers(self):
        params = list(chain(self.encoder.parameters(), self.decoder.parameters()))
        if self.quantizer is not None:
            params.extend(self.quantizer.parameters())
        return torch.optim.Adam(params, lr=self.hparams.lr)
