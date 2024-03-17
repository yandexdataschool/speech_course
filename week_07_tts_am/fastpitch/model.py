
import torch
from torch import nn as nn

from week_07_tts_am.fastpitch.common.layers import TemporalPredictor
from week_07_tts_am.fastpitch.common.utils import DeviceGetterMixin
from week_07_tts_am.fastpitch.common.utils import regulate_len
from week_07_tts_am.fastpitch.data import FastPitchBatch, SymbolsSet
from week_07_tts_am.fastpitch.hparams import HParamsFastpitch
from week_07_tts_am.fastpitch.common.transformer import FFTransformer


class FastPitch(nn.Module, DeviceGetterMixin):
    def __init__(self, hparams: HParamsFastpitch):
        <YOUR CODE HERE>
