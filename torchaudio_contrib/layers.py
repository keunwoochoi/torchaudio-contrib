import torch
import math
import torch.nn as nn

from .functional import stft, complex_norm, \
    create_mel_filter, phase_vocoder, apply_filterbank


class _ModuleNoStateBuffers(nn.Module):
    """
    Extension of nn.Module that removes buffers
    from state_dict.
    """

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        ret = super(_ModuleNoStateBuffers, self).state_dict(
            destination, prefix, keep_vars)
        for k in self._buffers:
            del ret[prefix + k]
        return ret

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # temporarily hide the buffers; we do not want to restore them

        buffers = self._buffers
        self._buffers = {}
        result = super(_ModuleNoStateBuffers, self)._load_from_state_dict(
            state_dict, prefix, *args, **kwargs)
        self._buffers = buffers
        return result


class STFT(_ModuleNoStateBuffers):
    """
    Compute the stft transform of a multi-channel signal or
    batch of multi-channel signals.

    Args:

        fft_len (int): FFT window size. Defaults to 2048.
        hop_len (int): Number audio of frames between stft columns.
            Defaults to fft_len // 4.
        frame_len (int): Size of stft window. Defaults to fft_len.
        window (Tensor): 1-D tensor. Defaults to Hann Window
            of size frame_len.
        pad (int): Amount of padding to apply to signal. Defaults to 0.
        pad_mode: padding method (see torch.nn.functional.pad).
            Defaults to "reflect".
        **kwargs: Other torch.stft parameters, see torch.stft for more details.

    """

    def __init__(self, fft_len=2048, hop_len=None, frame_len=None,
                 window=None, pad=0, pad_mode="reflect", **kwargs):

        super(STFT, self).__init__()

        # Get default values, window so it can be registered as buffer
        self.fft_len, self.hop_len, window = self._stft_defaults(
            fft_len, hop_len, frame_len, window)

        self.pad = pad
        self.pad_mode = pad_mode
        self.kwargs = kwargs

        self.register_buffer('window', window)

    def _stft_defaults(self, fft_len, hop_len, frame_len, window):
        """
        Handle default values for STFT.
        """
        hop_len = fft_len // 4 if hop_len is None else hop_len

        if window is None:
            length = fft_len if frame_len is None else frame_len
            window = torch.hann_window(length)
        if not isinstance(window, torch.Tensor):
            raise TypeError('window must be a of type torch.Tensor')

        return fft_len, hop_len, window

    def forward(self, signal):
        """
        Args:
            signal (Tensor): (channel, time) or (batch, channel, time).

        Returns:
            spect (Tensor): (channel, time, freq, complex)
                or (batch, channel, time, freq, complex).
        """

        spect = stft(signal, self.fft_len, self.hop_len, window=self.window,
                     pad=self.pad, pad_mode=self.pad_mode, **self.kwargs)

        return spect

    def __repr__(self):
        param_str = '(fft_len={}, hop_len={}, frame_len={})'.format(
            self.fft_len, self.hop_len, self.window.size(0))
        return self.__class__.__name__ + param_str


class ComplexNorm(nn.Module):
    """
    Wrap torchaudio_contrib.complex_norm in an nn.Module.
    """

    def __init__(self, power=1.0):
        super(ComplexNorm, self).__init__()
        self.power = power

    def forward(self, stft):
        return complex_norm(stft, self.power)

    def __repr__(self):
        return self.__class__.__name__ + '(power={})'.format(self.power)


class ApplyFilterbank(_ModuleNoStateBuffers):
    """
    Applies a filterbank transform.
    """

    def __init__(self, filterbank):
        super(ApplyFilterbank, self).__init__()
        self.register_buffer('filterbank', filterbank)

    def forward(self, spect):
        """
        Args:
            spect (Tensor): (channel, time, freq) or (batch, channel, time, freq).

        Returns:
            (Tensor): freq -> filterbank.size(0)
        """
        return apply_filterbank(spect, self.filterbank)


class Filterbank(object):
    """
    Base class for providing a filterbank matrix.
    """

    def __init__(self):
        super(Filterbank, self).__init__()

    def get_filterbank(self):
        raise NotImplementedError


class MelFilterbank(Filterbank):
    """
    Provides a filterbank matrix to convert a spectrogram into a mel frequency spectrogram.

    Args:
        num_bands (int): number of mel bins. Defaults to 128.
        sample_rate (int): sample rate of audio signal. Defaults to 22050.
        min_freq (float): minimum frequency. Defaults to 0.
        max_freq (float, optional): maximum frequency. Defaults to sample_rate // 2.
        num_bins (int, optional): number of filter banks from stft.
            Defaults to 2048//2 + 1.
        htk (bool, optional): use HTK formula instead of Slaney. Defaults to False.
    """

    def __init__(self, num_bands=128, sample_rate=22050,
                 min_freq=0.0, max_freq=None, num_bins=1025, htk=False):

        super(MelFilterbank, self).__init__()

        self.num_bands = num_bands
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.max_freq = max_freq if max_freq else sample_rate // 2
        self.num_bins = num_bins
        self.htk = htk

    def to_hertz(self, mel):
        """
        Converting mel values into frequency
        """
        if self.htk:
            return 700. * (10**(mel / 2595.) - 1.)

        mel = torch.as_tensor(mel).float()

        f_min = 0.0
        f_sp = 200.0 / 3
        hz = f_min + f_sp * mel

        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0

        return torch.where(mel >= min_log_mel, min_log_hz *
                           torch.exp(logstep * (mel - min_log_mel)), hz)

    def from_hertz(self, hz):
        """
        Converting frequency into mel values
        """
        if self.htk:
            return 2595. * torch.log10(torch.tensor(1.) + (hz / 700.))

        hz = torch.as_tensor(hz).float()
        f_min = 0.0
        f_sp = 200.0 / 3

        mel = (hz - f_min) / f_sp

        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0

        return torch.where(hz >= min_log_hz, min_log_mel +
                           torch.log(hz / min_log_hz) / logstep, mel)

    def get_filterbank(self):
        return create_mel_filter(
            num_bands=self.num_bands,
            sample_rate=self.sample_rate,
            min_freq=self.min_freq,
            max_freq=self.max_freq,
            num_bins=self.num_bins,
            to_hertz=self.to_hertz,
            from_hertz=self.from_hertz)

    def __repr__(self):
        param_str = '(num_bands={}, sample_rate={}, min_freq={}, max_freq={})'.format(
            self.num_bands, self.sample_rate, self.min_freq, self.max_freq)
        return self.__class__.__name__ + param_str


class StretchSpecTime(_ModuleNoStateBuffers):
    """
    Stretch stft in time without modifying pitch for a given rate.

    Args:

        rate (float): rate to speed up or slow down by.
        hop_len (int): Number audio of frames between STFT columns.
            Defaults to 512.
        num_bins (int, optional): number of filter banks from stft.
            Defaults to 1025.
    """

    def __init__(self, rate, hop_len=512, num_bins=1025):

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


def Spectrogram(fft_len=2048, hop_len=None, frame_len=None,
                window=None, pad=0, pad_mode="reflect", power=1., **kwargs):
    """
    Get spectrogram module.

    Args:

        fft_len (int): FFT window size. Defaults to 2048.
        hop_len (int): Number audio of frames between STFT columns.
            Defaults to fft_len // 4.
        frame_len (int): Size of stft window. Defaults to fft_len.
        window (Tensor): 1-D tensor.
            Defaults to Hann Window of size frame_len.
        pad (int): Amount of padding to apply to signal. Defaults to 0.
        pad_mode: padding method (see torch.nn.functional.pad).
            Defaults to "reflect".
        power (float): Exponent of the magnitude. Defaults to 1.
        **kwargs: Other torch.stft parameters, see torch.stft for more details.
    """
    return nn.Sequential(STFT(fft_len, hop_len, frame_len,
                              window, pad, pad_mode, **kwargs), ComplexNorm(power))


def Melspectrogram(num_bands=128, sample_rate=22050, min_freq=0.0,
                   max_freq=None, num_bins=None, htk=False, mel_filterbank=None, **kwargs):
    """
    Get melspectrogram module.

    Args:
        num_bands (int): number of mel bins. Defaults to 128.
        sample_rate (int): sample rate of audio signal. Defaults to 22050.
        min_freq (float): minimum frequency. Defaults to 0.
        max_freq (float, optional): maximum frequency. Defaults to sample_rate // 2.
        num_bins (int, optional): number of filter banks from stft.
            Defaults to fft_len//2 + 1 if 'fft_len' in kwargs else 1025.
        htk (bool, optional): use HTK formula instead of Slaney. Defaults to False.
        mel_filterbank (class, optional): MelFilterbank class to build filterbank matrix
        **kwargs: torchaudio_contrib.Spectrogram parameters.
    """
    fft_len = kwargs.get('fft_len', None)
    num_bins = fft_len // 2 + 1 if fft_len else 1025

    # Check if custom MelFilterbank is passed
    if mel_filterbank is None:
        mel_filterbank = MelFilterbank

    mel_fb_matrix = mel_filterbank(
        num_bands, sample_rate, min_freq, max_freq, num_bins, htk).get_filterbank()

    return nn.Sequential(*Spectrogram(power=2., **kwargs),
                         ApplyFilterbank(mel_fb_matrix))
