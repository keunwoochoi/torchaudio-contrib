import torch
import math
import torch.nn as nn

from .functional import stft_defaults, _stft, complex_norm, create_mel_filter, phase_vocoder


class _ModuleNoStateBuffers(nn.Module):
    """
    Extension of nn.Module
    """

    def __init__(self, buffs):
        super(_ModuleNoStateBuffers, self).__init__()
        for k, v in buffs.items():
            self.register_buffer(k, v)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        ret = super(_ModuleNoStateBuffers, self).state_dict(
            destination, prefix, keep_vars)
        for k in self._buffers.keys():
            del ret[prefix+k]
        return ret


class STFT(_ModuleNoStateBuffers):
    """
    Compute the stft transform of a multi-channel signal or
    batch of multi-channel signals.

    Args:

        n_fft (int): FFT window size. Defaults to 2048.
        hop_length (int): Number audio of frames between stft columns. Defaults to n_fft // 4.
        len_win (int): Size of stft window. Defaults to n_fft.
        window (Tensor): 1-D tensor. Defaults to Hanning Window of size len_win.
        pad (int): Amount of padding to apply to signal. Defaults to 0.
        pad_mode: padding method (see torch.nn.functional.pad). Defaults to "reflect".
        **kwargs: Other torch.stft parameters, see torch.stft for more details.

    """

    def __init__(self, n_fft=2048, hop_length=None, len_win=None,
                 window=None, pad=0, pad_mode="reflect", **kwargs):

        # Get default values, window so it can be registered as buffer
        self.n_fft, self.hop_length, window = stft_defaults(
            n_fft, hop_length, len_win, window)

        self.pad = pad
        self.pad_mode = pad_mode
        self.kwargs = kwargs

        super(STFT, self).__init__({'window': window})

    def forward(self, x):
        """
        Args:
            x (Tensor): (channel, signal) or (batch, channel, signal).

        Returns:
            stft_out (Tensor): (channel, time, freq, complex) or (batch, channel, time, freq, complex).
        """

        # use registered window so have to use _stft
        stft_out = _stft(x, self.n_fft, self.hop_length, window=self.window, pad=self.pad,
                         pad_mode=self.pad_mode, **self.kwargs)

        return stft_out

    def __repr__(self):
        param_str = '(n_fft={}, hop_length={}, len_win={}, pad={})'.format(
            self.n_fft, self.hop_length, self.window.size(0), self.pad)
        return self.__class__.__name__ + param_str


class ComplexNorm(nn.Module):
    """
    Wrap complex_norm in a nn.Module.
    """

    def __init__(self, power=1.0):
        super(ComplexNorm, self).__init__()
        self.power = power

    def forward(self, x):
        return complex_norm(x, self.power)


class Filterbank(_ModuleNoStateBuffers):
    def __init__(self):
        super(Filterbank, self).__init__({'fb': self._build_fb()})

    def _build_fb(self):
        raise NotImplementedError

    def forward(self, x):
        """
        Args:
            x (Tensor): (channel, time, freq) or (batch, channel, time, freq).

        Returns:
            fb_out (Tensor): freq -> fb.size(0)
        """
        return torch.matmul(x.transpose(2, 3), self.fb).transpose(2, 3)


class MelFilterbank(Filterbank):
    """
    Convert a spectrogram into a mel frequency spectrogram using
    a filterbank.

    Args:
        n_mels (int): number of mel bins. Defaults to 128.
        sr (int): sample rate of audio signal. Defaults to 44100.
        f_max (float, optional): maximum frequency. Defaults to sr // 2.
        f_min (float): minimum frequency. Defaults to 0.
        n_stft (int, optional): number of filter banks from stft. Defaults to 2048//2 + 1.
    """

    def __init__(self, n_mels=128, sr=44100, f_min=0.0, f_max=None, n_stft=None):

        self.n_stft = n_stft
        self.n_mels = n_mels
        self.sr = sr
        self.f_max = f_max
        self.f_min = f_min

        super(MelFilterbank, self).__init__()

    def _build_fb(self):
        return create_mel_filter(
            n_mels=self.n_mels,
            sr=self.sr,
            f_min=self.f_min,
            f_max=self.f_max,
            n_stft=self.n_stft)

    def __repr__(self):
        param_str = '(n_mels={}, sr={}, f_min={}, f_max={})'.format(
            self.n_mels, self.sr, self.f_min, self.f_max)
        return self.__class__.__name__ + param_str


class StretchSpecTime(_ModuleNoStateBuffers):
    """
    Stretch stft in time without modifying pitch for a given rate.

    Args:

        rate (float): rate to speed up or slow down by.
        hop_length (int): Number audio of frames between STFT columns. Defaults to 512.
        n_stft (int, optional): number of filter banks from stft. Defaults to 1025.
    """

    def __init__(self, rate, hop_length=512, n_stft=1025):

        self.rate = rate

        phi_advance = torch.linspace(
            0, math.pi * hop_length, n_stft)[..., None]

        super(StretchSpecTime, self).__init__({'phi_advance': phi_advance})

    def forward(self, x):
        return phase_vocoder(x, self.rate, self.phi_advance)

    def __repr__(self):
        param_str = '(rate={})'.format(self.rate)
        return self.__class__.__name__ + param_str


def Spectrogram(n_fft=2048, hop_length=None, len_win=None,
                window=None, pad=0, pad_mode="reflect", power=1., **kwargs):
    """
    Get spectrogram module.

    Args:

        n_fft (int): FFT window size. Defaults to 2048.
        hop_length (int): Number audio of frames between STFT columns. Defaults to n_fft // 4.
        len_win (int): Size of stft window. Defaults to n_fft.
        window (Tensor): 1-D tensor. Defaults to Hanning Window of size len_win.
        pad (int): Amount of padding to apply to signal. Defaults to 0.
        pad_mode: padding method (see torch.nn.functional.pad). Defaults to "reflect".
        power (float): What power to normalize to.
        **kwargs: Other torch.stft parameters, see torch.stft for more details.
    """
    return nn.Sequential(STFT(n_fft, hop_length, len_win,
                              window, pad, pad_mode, **kwargs), ComplexNorm(power))


def Melspectrogram(n_mels=128, sr=44100, f_min=0.0, f_max=None, n_stft=None, **kwargs):
    """
    Get melspectrogram module.

    Args:
        n_mels (int): number of mel bins.
        sr (int): sample rate of audio signal.
        f_max (float, optional): maximum frequency. Defaults to sr // 2.
        f_min (float): minimum frequency. Defaults to 0.
        n_stft (int, optional): number of filter banks from stft. Defaults to n_fft//2 + 1 if 'n_fft' in kwargs else 1025.
        **kwargs: torchaudio_contrib.Spectrogram parameters.
    """
    n_fft = kwargs.get('n_fft', None)
    if n_fft:
        n_stft = n_fft//2 + 1
    return nn.Sequential(*Spectrogram(**kwargs),
                         MelFilterbank(n_mels, sr, f_min, f_max, n_stft))
