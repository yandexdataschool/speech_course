# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import torch
from torch import nn as nn
import torch.nn.functional as F

SUPPORTED_CONDITION_TYPES = ['add', 'concat', 'layernorm']


def check_support_condition_types(condition_types):
    for tp in condition_types:
        if tp not in SUPPORTED_CONDITION_TYPES:
            raise ValueError(f'Unknown conditioning type {tp}')


class ConditionalLayerNorm(torch.nn.LayerNorm):
    """
    This module is used to condition torch.nn.LayerNorm.
    If we don't have any conditions, this will be a normal LayerNorm.
    """

    def __init__(self, hidden_dim, condition_dim=None, condition_types=None):
        condition_types = condition_types or []
        check_support_condition_types(condition_types)
        self.condition = "layernorm" in condition_types
        super().__init__(hidden_dim, elementwise_affine=not self.condition)

        if self.condition:
            self.cond_weight = torch.nn.Linear(condition_dim, hidden_dim)
            self.cond_bias = torch.nn.Linear(condition_dim, hidden_dim)
            self.init_parameters()

    def init_parameters(self):
        torch.nn.init.constant_(self.cond_weight.weight, 0.0)
        torch.nn.init.constant_(self.cond_weight.bias, 1.0)
        torch.nn.init.constant_(self.cond_bias.weight, 0.0)
        torch.nn.init.constant_(self.cond_bias.bias, 0.0)

    def forward(self, inputs, conditioning=None):
        inputs = super().forward(inputs)

        # Normalize along channel
        if self.condition:
            if conditioning is None:
                raise ValueError(
                    """You should add additional data types as conditions (e.g. speaker id or reference audio)
                                 and define speaker_encoder in your config."""
                )

            inputs = inputs * self.cond_weight(conditioning)
            inputs = inputs + self.cond_bias(conditioning)

        return inputs


class ConditionalInput(torch.nn.Module):
    """
    This module is used to condition any model inputs.
    If we don't have any conditions, this will be a normal pass.
    """

    def __init__(self, hidden_dim, condition_dim, condition_types=None):
        condition_types = condition_types or []
        check_support_condition_types(condition_types)
        super().__init__()
        self.support_types = ['add', 'concat']
        self.condition_types = [tp for tp in condition_types if tp in self.support_types]
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim

        if 'add' in self.condition_types and condition_dim != hidden_dim:
            self.add_proj = torch.nn.Linear(condition_dim, hidden_dim)

        if 'concat' in self.condition_types:
            self.concat_proj = torch.nn.Linear(hidden_dim + condition_dim, hidden_dim)

    def forward(self, inputs, conditioning=None):
        """
        Args:
            inputs (torch.tensor): B x seq_len x hidden_dim tensor.
            conditioning (torch.tensor): B x (1 or seq_len) x condition_dim conditioning embedding.
        """
        if len(self.condition_types) > 0:
            if conditioning is None:
                raise ValueError(
                    """You should add additional data types as conditions (e.g. speaker id or reference audio)
                                 and define speaker_encoder in your config."""
                )

            if 'add' in self.condition_types:
                if self.condition_dim != self.hidden_dim:
                    conditioning = self.add_proj(conditioning)
                inputs = inputs + conditioning

            if 'concat' in self.condition_types:
                if conditioning.shape[1] == 1:
                    conditioning = conditioning.repeat(1, inputs.shape[1], 1)
                elif conditioning.shape[1] != inputs.shape[1]:
                    raise ValueError('Conditioning seq_len must be either 1 or equal to input seq_len')
                inputs = torch.cat([inputs, conditioning], dim=2)
                inputs = self.concat_proj(inputs)

        return inputs


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear', batch_norm=False):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)
        self.norm = torch.nn.BatchNorm1D(out_channels) if batch_norm else None

        torch.nn.init.xavier_uniform_(
            self.conv.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        if self.norm is None:
            return self.conv(signal)
        else:
            return self.norm(self.conv(signal))


class ConvReLUNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.0, condition_dim=None, condition_types=None):
        super(ConvReLUNorm, self).__init__()
        condition_types = condition_types or []
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size,
                                    padding=(kernel_size // 2))
        self.norm = ConditionalLayerNorm(
            out_channels,
            condition_dim=condition_dim,
            condition_types=condition_types
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, signal, conditioning=None):
        out = F.relu(self.conv(signal))
        out = self.norm(out.transpose(1, 2), conditioning).transpose(1, 2)
        return self.dropout(out)


class TemporalPredictor(nn.Module):
    '''
    Predicts a single float per each temporal location
    '''
    def __init__(self, input_size, filter_size, kernel_size, dropout, n_layers=2, condition_types=None):
        super(TemporalPredictor, self).__init__()
        condition_types = condition_types or []

        self.layers = nn.ModuleList([
            ConvReLUNorm(
                input_size if i == 0 else filter_size,
                filter_size,
                kernel_size=kernel_size,
                dropout=dropout,
                condition_dim=input_size,
                condition_types=condition_types,
            )
            for i in range(n_layers)
        ])
        self.fc = nn.Linear(filter_size, 1, bias=True)

    def forward(self, enc_out, enc_out_mask, conditioning=None):
        out = enc_out * enc_out_mask
        out = out.transpose(1, 2)

        for layer in self.layers:
            out = layer(out, conditioning=conditioning)

        out = out.transpose(1, 2)
        out = self.fc(out) * enc_out_mask
        return out.squeeze(-1)
    