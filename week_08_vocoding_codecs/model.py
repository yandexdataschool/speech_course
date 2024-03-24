from typing import Optional
from itertools import chain

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

from vector_quantization import Perplexity, VectorQuantizer, ResidualVectorQuantizer


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=1,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)


class MNISTEncoderDecoder(L.LightningModule):
    def __init__(self, quantizer: Optional[nn.Module] = None, vq_loss: Optional[nn.Module] = None):
        super().__init__()

        self.encoder = Encoder(in_channels=1, num_hiddens=16, num_residual_layers=2, num_residual_hiddens=4)
        self.decoder = Decoder(in_channels=16, num_hiddens=16, num_residual_layers=2, num_residual_hiddens=4)

        self.quantizer = quantizer
        self.vq_loss_fn = vq_loss

        self.reconstr_loss_fn = nn.MSELoss()
        if self.quantizer is None:
            self.perplexity = lambda x: torch.tensor(-1.)
        else:
            self.perplexity = Perplexity(n_codecs=self.quantizer.codebook_size)

    def forward(self, x):
        encoded = self.encoder(x)
        quantized = self.quantizer(encoded)
        decoded = self.decoder(quantized)
        return decoded

    def training_step(self, batch, batch_idx):
        pictures_batch, idx = batch
        if self.quantizer is None:
            loss = self.training_step_no_quantizer(pictures_batch)
        else:
            loss = self.training_step_with_quantizer(pictures_batch)

        self.log("train/loss", loss.item(), prog_bar=True)
        return loss

    def training_step_no_quantizer(self, pictures_batch):
        encoded = self.encoder(pictures_batch)
        predicted = self.decoder(encoded)

        loss = self.reconstr_loss_fn(predicted, pictures_batch)

        return loss

    def training_step_with_quantizer(self, pictures_batch):
        # Your code here
        raise NotImplementedError("TODO: assignment")

        # ^^^^^^^^^^^^^^

        return loss

    def on_after_backward(self):
        # Sanity checks
        for p in self.encoder.parameters():
            if p.grad is None:
                raise RuntimeError("Error, gradinent in self.encoder after backward is None")
        if self.quantizer is not None:
            for p in self.quantizer.parameters():
                if p.grad is None:
                    raise RuntimeError("Error, gradinent in self.quantizer after backward is None")
        for p in self.decoder.parameters():
            if p.grad is None:
                raise RuntimeError("Error, gradinent in self.decoder after backward is None")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pic, idx = batch

        encoded = self.encoder(pic)
        if self.quantizer:
            indices = self.quantizer.encode(encoded)
            quantized = self.quantizer.decode(indices)
        else:
            indices = None
            quantized = encoded
        assert encoded.shape == quantized.shape
        predicted = self.decoder(quantized)

        reconstr_loss = self.reconstr_loss_fn(predicted, pic)
        vq_loss = self.vq_loss_fn(encoded, quantized) if self.vq_loss_fn else torch.tensor(0.)
        loss = reconstr_loss + vq_loss

        self.log("val/reconstr_loss", reconstr_loss.item(), on_epoch=True)
        self.log("val/vq_loss", vq_loss.item(), on_epoch=True)
        self.log("val/loss", loss.item(), on_epoch=True, prog_bar=True)

        perp = self.perplexity(indices)
        self.log("val/perplexity", perp.item(), on_epoch=True, prog_bar=True)

        encoded_norm = encoded.norm(dim=1).mean(dim=(0, 1, 2))
        quantized_norm = quantized.norm(dim=1).mean(dim=(0, 1, 2))
        codebook_vectors_norm = None
        if isinstance(self.quantizer, VectorQuantizer):
            codebook_vectors_norm = self.quantizer.codebook.weight.norm(dim=1).mean(dim=0)
        elif isinstance(self.quantizer, ResidualVectorQuantizer):
            codebook_vectors_norm = self.quantizer.codebooks[0].codebook.weight.norm(dim=1).mean(dim=0)

        self.log("norm/encoded", encoded_norm.item())
        self.log("norm/quantized", quantized_norm.item())
        if codebook_vectors_norm is not None:
            self.log("norm/codebook_vectors", codebook_vectors_norm.item())

        if batch_idx == 0:
            orig_img = self.plot_batch(pic)
            restored_img = self.plot_batch(predicted)
            self.logger.experiment.add_image("orig", orig_img, global_step=self.trainer.current_epoch)
            self.logger.experiment.add_image("restored", restored_img, global_step=self.trainer.current_epoch)

        return loss

    def configure_optimizers(self):
        params = list(chain(self.encoder.parameters(), self.decoder.parameters()))
        if self.quantizer is not None:
            params.extend(self.quantizer.parameters())
        
        opt = torch.optim.Adam(params, lr=0.003)
        return opt
    
    def plot_batch(self, pic):
        grid = make_grid(pic.cpu(), normalize=True)
        return grid
