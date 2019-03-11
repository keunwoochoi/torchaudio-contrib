

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import spectrogram, create_mel_filter


class Spectrogram(nn.Module):
    """
    Module that outputs the spectrogram
    of an audio signal with shape (batch, channel, time_hop, frequency_bins).

    Its implemented as a layer so that the computation can be faster (done dynamically
    on GPU) and no need to store the transforms. More information:
        - https://github.com/keunwoochoi/kapre
        - https://arxiv.org/pdf/1706.05781.pdf

    Args:
     * hop: int > 0
       -  Hop length between frames in sample,  should be <= n_fft.
       -  Default: None (in which case n_fft // 4 is used)
     * n_fft: int > 0 
       -  Size of the fft.
       -  Default: 2048
     * pad: int >= 0
       -  Amount of two sided zero padding to apply.
       -  Default: 0
     * window: torch.Tensor,
       -  Windowing used in the stft.
       -  Default: None (in which case torch.hann_window(n_fft) is used)
     * sr: int > 0
       -  Sampling rate of the audio signal. This may not be the same in all samples (?)
       -  Default: 44100
     * power: int = 1,2,None
       -  Power to normalize the spectrogram to, make None to work with complex stft
       -  Default: 1
     * kwargs: 
       -  Any named arguments to be passed to the stft

    """

    def __init__(self, hop=None, n_fft=2048, pad=0, window=None, sr=44100, **kwargs):

        super(Spectrogram, self).__init__()

        if window is None:
            window = torch.hann_window(n_fft)

        self.window = self._build_window(window)
        self.hop = n_fft // 4 if hop is None else hop
        self.n_fft = n_fft
        self.pad = pad

        # Not all samples will have the same sr
        self.sr = sr
        self.kwargs = kwargs

    def _build_window(self, window):
        if window is None:
            window = torch.hann_window(n_fft)
        if not isinstance(window, torch.Tensor):
            raise TypeError('window must be a of type torch.Tensor')
        # In order for the window to be added as one of the Module's
        # parameters it has to be a nn.Parameter
        return nn.Parameter(window, requires_grad=False)

    def _out_seq_dim(self, arr):
        return arr//self.hop+1

    def forward(self, x, lengths=None):
        """
        Input Tensor shape -> (batch, channel, signal)
        Output Tensor shape -> (batch, channel, freq, time) or
            (batch, channel, freq, time, complex) if power=None
        """

        if self.pad > 0:
            with torch.no_grad():
                x = F.pad(x, (self.pad, self.pad), "constant")

        spec = spectrogram(x,
                    n_fft=self.n_fft,
                    hop=self.hop,
                    window=self.window,
                    **self.kwargs)

        return spec


class Melspectrogram(Spectrogram):
    """
    Module that outputs the mel-spectrogram (transform on the spectrogram
    to better represent human perception) of an audio signal.

    Args:
     * n_mels: int > 0 
       -  Number of mel bands.
       -  Default: 128
     * sr: int > 0
       -  Sampling rate of the audio signal. This may not be the same in all samples (?)
       -  Default: 44100
    * f_min: float > 0
       -  Lowest freq. in Hz
       -  Default: 0.
    * f_max: float > 0
       -  Highest freq. in Hz
       -  Default: None (then use sr / 2.0)
    * args: 
       -  Positional arguments for Spectrogram
    * kwargs: 
       -  Keyword arguments for Spectrogram
    """

    def __init__(self, n_mels=128, sr=44100, f_min=0.0, f_max=None, *args, **kwargs):

        super(Melspectrogram, self).__init__(*args, **kwargs)
        self.sr = sr
        self.n_mels = n_mels
        self.mel_fb, self.mel_freq_vals = self._build_filter(sr, f_min, f_max)

    def _build_filter(self, sr, f_min, f_max):
        # Get the mel filter matrix and the mel frequency values
        mel_fb, mel_f = create_mel_filter(
            self.n_fft//2 + 1,
            sr,
            n_mels=self.n_mels,
            f_min=f_min,
            f_max=f_max)
        # Cast filter matrix as nn.Parameter so it's loaded on model's device
        return nn.Parameter(mel_fb, requires_grad=False), mel_f

    def forward(self, x):
        """
        Input Tensor shape -> (batch, channel, signal)
        Output Tensor shape -> (batch, channel, mel_freq, time)
        """

        spec = super(Melspectrogram, self).forward(x)
        spec = torch.matmul(spec.transpose(2, 3), self.mel_fb).transpose(2, 3)
        return spec
