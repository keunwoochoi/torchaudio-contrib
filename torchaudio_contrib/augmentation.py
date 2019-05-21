import math
import torch

import torch.nn as nn
from .layers import _ModuleNoStateBuffers, STFT
from .functional import phase_vocoder


class StretchSpecTime(_ModuleNoStateBuffers):
    """
    Stretch stft in time without modifying pitch for a given rate.

    Args:
        rate (float): rate to speed up or slow down by. Defaults to 1.
        hop_len (int): Number audio of frames between STFT columns.
            Defaults to 512.
        num_bins (int, optional): number of filter banks from stft.
            Defaults to 1025.
    """

    def __init__(self, rate=1., hop_len=512, num_bins=1025):

        super(StretchSpecTime, self).__init__()

        self.rate = rate
        phi_advance = torch.linspace(
            0, math.pi * hop_len, num_bins)[..., None]

        self.register_buffer('phi_advance', phi_advance)

    def forward(self, spect, rate=None):
        if rate is None:
            rate = self.rate
        return phase_vocoder(spect, rate, self.phi_advance)

    def __repr__(self):
        param_str = '(rate={})'.format(self.rate)
        return self.__class__.__name__ + param_str


class GaussianNoise(_ModuleNoStateBuffers):
    """
    Apply gaussian noise to a spectrogram.

    Args:
        scale (float or tensor): standard deviation of the distribution.
    """

    def __init__(self, scale=1.):
        super(GaussianNoise, self).__init__()
        self.scale = torch.as_tensor(scale)
        self.loc = torch.tensor(0.)

    def forward(self, spect, scale=None):
        if scale is None:
            scale = self.scale
        else:
            scale = torch.as_tensor(scale)

        with torch.no_grad():

            noise = torch.normal(
                self.loc.expand(
                    spect.shape), scale.expand(
                    spect.shape))
            return spect + noise

    def __repr__(self):
        param_str = '(scale={})'.format(self.scale)
        return self.__class__.__name__ + param_str


def apply_masking(spect, max_value, mask_value, axis):

    assert axis in [2, 3]

    v = torch.rand(spect.shape[:2]) * max_value
    v_0 = torch.rand(spect.shape[:2]) * (spect.size(axis) - v)

    mask_start = (v_0.long()).unsqueeze(-1)
    mask_end = (v_0.long() + v.long()).unsqueeze(-1)

    mask = torch.arange(
        0, spect.size(axis)).repeat(
        spect.size(0), spect.size(1), 1)

    spect = spect.transpose(2, axis)

    # per (batch, channel) mask
    spect[(mask >= mask_start) & (mask < mask_end)] = mask_value
    spect = spect.transpose(2, axis)

    return spect


def apply_masking_batch(spect, max_value, mask_value, axis):

    v = torch.rand(1) * max_value
    v_0 = torch.rand(1) * (spect.size(axis) - v)

    mask_start = (v_0.long()).squeeze()
    mask_end = (v_0.long() + v.long()).squeeze()

    if axis == 2:
        spect[:, :, mask_start:mask_end] = mask_value
    elif axis == 3:
        spect[:, :, :, mask_start:mask_end] = mask_value
    else:
        raise ValueError('Only Frequency and Time masking is supported')

    return spect


class _AxisMasking(_ModuleNoStateBuffers):

    def __init__(self, max_value, axis, same_batch):

        super(_AxisMasking, self).__init__()
        self.max_value = max_value
        self.axis = axis

        if same_batch:
            self.masking = apply_masking_batch
        else:
            self.masking = apply_masking

    def forward(self, spect, mask_value=0):
        return self.masking(spect, self.max_value, mask_value, self.axis)


class FrequencyMasking(_AxisMasking):
    """
    Apply masking in the frequency domain.

    Args:
        max_freq (int): maximum possible length of the mask.
            Uniformly sampled from [0, max_freq).
        same_batch (bool): weather to apply the same mask to all
            the examples/channels in the batch. Defaults to False.
    """

    def __init__(self, max_freq, same_batch=False):
        super(FrequencyMasking, self).__init__(max_freq, 2, same_batch)


class TimeMasking(_AxisMasking):
    """
    Apply masking in the time domain.

    Args:
        max_time (int): maximum possible length of the mask.
            Uniformly sampled from [0, max_time).
        same_batch (bool): weather to apply the same mask to all
            the examples/channels in the batch. Defaults to False.
    """

    def __init__(self, max_time, same_batch=False):
        super(TimeMasking, self).__init__(max_time, 3, same_batch)
