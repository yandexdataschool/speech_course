from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


@dataclass
class SampledArray:
    value: np.ndarray
    t1: float  # center of the first frame
    step: float  # sampling step in seconds

    def n_samples(self):
        return self.value.shape[0]

    def time_to_sample(self, t):
        return round((t - self.t1) / self.step)

    def sample_to_time(self, f):
        return self.t1 + f * self.step

    def end(self):
        return self.t1 + (len(self.value)-1) * self.step

    def timeline(self):
        return np.linspace(self.t1, self.end(), num=len(self.value))

    def plot(self, *args, **kwargs):
        pd.Series(self.value, index=self.timeline()).plot(*args, **kwargs)


def resample(a: SampledArray, b: SampledArray, kind='nearest') -> SampledArray:
    '''
    Resamples values of a to the timeline of b
    '''
    x, y = a.timeline(), a.value
    inter_f = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]), kind=kind)
    return SampledArray(
        value=inter_f(b.timeline()),
        t1=b.t1,
        step=b.step
    )