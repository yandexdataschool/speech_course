import json

import torch
import torch.nn as nn

from .env import AttrDict
from .model import Encoder, Generator, Quantizer


class CodecModel(nn.Module):
    def __init__(self, config_path, ckpt_path, sample_rate=16000):
        super(CodecModel, self).__init__()
        self.sample_rate = sample_rate

        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
        with open(config_path) as f:
            data = f.read()
        json_config = json.loads(data)
        self.h = AttrDict(json_config)

        self.quantizer = Quantizer(self.h)
        self.generator = Generator(self.h)
        self.encoder = Encoder(self.h)

        self.generator.load_state_dict(ckpt['generator'])
        self.quantizer.load_state_dict(ckpt['quantizer'])
        self.encoder.load_state_dict(ckpt['encoder'])

    def decode(self, x, spkr):
        return self.generator(self.quantizer.embed(x), spkr)

    def encode(self, x):
        batch_size = x.size(0)
        c = self.encoder(x.unsqueeze(1))
        q, loss_q, c = self.quantizer(c)
        c = [code.reshape(batch_size, -1) for code in c]
        return torch.stack(c, -1)  # N, T, 4
