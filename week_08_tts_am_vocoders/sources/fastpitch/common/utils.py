from dataclasses import fields as dataclass_fields
from typing import Callable, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def assess_pitch_stats(local_dataset, pitch_extractor) -> (float, float):
    means = []
    stds = []
    for i in tqdm(range(len(local_dataset))):
        pitch = pitch_extractor(local_dataset[i], normalize_pitch=False)
        means.append(pitch[pitch > 0].mean())
        stds.append(pitch[pitch > 0].std())
        
    ids = np.where(~np.isnan(means))[0]
    mean_pitch = np.mean(np.array(means)[ids])

    ids = np.where(~np.isnan(stds))[0]
    mean_std = np.mean(np.array(stds)[ids])

    return mean_pitch, mean_std


def convert_to_tensor(rows: list[dict], field: str, to_torch_type: Callable[[np.array], torch.Tensor]):
    return [to_torch_type(r[field]) for r in rows]


def pad_tensor_list(tensor_list: list[torch.Tensor]) -> (torch.Tensor, torch.Tensor):
    lengths = torch.LongTensor([tensor.shape[0] for tensor in tensor_list])
    padded_tensor = torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True, padding_value=0)
    return padded_tensor, lengths


class ToDeviceMixin(object):
    def to(self, device):
        for field in dataclass_fields(self):
            value = getattr(self, field.name)
            if callable(getattr(value, 'to', None)):
                setattr(self, field.name, value.to(device))
        return self
    

class DeviceGetterMixin:
    """Mixin to add `device` property to a torch model."""

    @property
    def device(self):
        return next(self.parameters()).device


def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max().long()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


@torch.jit.script
def regulate_len(durations: torch.Tensor, enc_out: torch.Tensor, paces: Optional[torch.Tensor] = None):
    '''
    Duplicates each encoder output according to its duration
    '''
    durations = durations.float()
    if paces is not None:
        reps = torch.round(durations / paces[:, None])
    else:
        reps = torch.round(durations)
    reps = reps.long()
    dec_lens = reps.sum(dim=1)

    enc_rep = pad_sequence(
        [torch.repeat_interleave(o, r, dim=0) for o, r in zip(enc_out, reps)], batch_first=True
    )
    return enc_rep, dec_lens, reps
