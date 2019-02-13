

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import spectrogram, create_mel_filter, _get_freq_values


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
       - Hop length between frames in sample,  should be <= n_fft.
       - Default: None (in which case n_fft // 4 is used)
     * n_fft: int > 0 
       - Size of the fft.
       - Default: 2048
     * pad: int >= 0
       - Amount of two sided zero padding to apply.
       - Default: 0
     * window: torch.Tensor,
       -  Windowing used in the stft.
       -  Default: None (in which case torch.hann_window(n_fft) is used)
     * sr: int > 0
       -  Sampling rate of the audio signal. This may not be the same in all samples (?)
       -  Default: 44100
     * spec_kwargs: 
       -  Any named arguments to be passed to the stft

    """

    def __init__(self, hop=None, n_fft=2048, pad=0, window=None, sr=44100, **spec_kwargs):
        
        super(Spectrogram, self).__init__()

        if window is None:
            window = torch.hann_window(n_fft)

        self.window = self._build_window(window)
        self.hop = n_fft // 4 if hop is None else hop
        self.n_fft = n_fft
        self.pad = pad

        # Not all samples will have the same sr
        self.freq_vals = _get_freq_values(n_fft, sr)
        self.sr = sr
        self.spec_kwargs = spec_kwargs


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
        If x is a padded tensor then lengths should have the 
        corresponding sequence length for every element in the batch.

        Input: (batch, channel, signal)
        Output:(batch, channel, time_hop, frequency_bins)
        """
        assert x.dim() == 3

        batch, channel, _ = x.size()
        x = x.reshape(batch*channel, -1) # (batch*channel, signal)

        if self.pad > 0:
            with torch.no_grad():
                x = F.pad(x, (self.pad, self.pad), "constant")

        spec = spectrogram(x,
            n_fft=self.n_fft,
            hop=self.hop, 
            window=self.window,
            **self.spec_kwargs)

        spec = spec.contiguous().view(batch, channel, -1, self.n_fft//2 + 1)
        
        if lengths is not None:            
            assert spec.size(0) == lengths.size(0)
            return spec, self._out_seq_dim(lengths)
        return spec


class Melspectrogram(Spectrogram):
    """
    Module that outputs the mel-spectrogram (transform on the spectrogram
    to better represent human perception) of an audio signal with output 
    shape (batch, channel, time_hop, frequency_bins).

    Its implemented as a layer so that the computation can be faster (done dynamically
    on GPU) and no need to store the transforms. More information:
        - https://github.com/keunwoochoi/kapre
        - https://arxiv.org/pdf/1706.05781.pdf
    
    Args:
     * hop: int > 0
       - Hop length between frames in sample,  should be <= n_fft.
       - Default: None (in which case n_fft // 4 is used)
     * n_mels: int > 0 
       - Number of mel bands.
       - Default: 2048
     * n_fft: int > 0 
       - Size of the fft.
       - Default: 2048
     * pad: int >= 0
       - Amount of two sided zero padding to apply.
       - Default: 0
     * window: torch.Tensor,
       -  Windowing used in the stft.
       -  Default: None (in which case torch.hann_window(n_fft) is used)
     * sr: int > 0
       -  Sampling rate of the audio signal. This may not be the same in all samples (?)
       -  Default: 44100
     * spec_kwargs: 
       -  Any named arguments to be passed to the stft

    """

    def __init__(self, hop=None, n_mels=128, n_fft=2048, pad=0, window=None, sr=44100, **spec_kwargs):
        
        super(Melspectrogram, self).__init__(hop, n_fft, pad, window, sr, **spec_kwargs)

        self.n_mels = n_mels
        self.mel_fb, self.mel_freq_vals = self._build_filter()

    def _build_filter(self):
        # Get the mel filter matrix and the mel frequency values
        mel_fb, mel_f = create_mel_filter(
                                    len(self.freq_vals),
                                    self.sr, 
                                    n_mels=self.n_mels)
        # Cast filter matrix as nn.Parameter so it's loaded on model's device 
        return nn.Parameter(mel_fb, requires_grad=False), mel_f

            
    def forward(self, x, lengths=None):

        spec = super(Melspectrogram, self).forward(x)
        spec = torch.matmul(spec, self.mel_fb)

        if lengths is not None:
            return spec, self._out_seq_dim(lengths)
        return spec



class MaskConv2d(nn.Conv2d):
    """
    Allow Conv2d to work with sequence data (or different height images).

    Expects (batch, channel, sequence, feature)
    """
    def __init__(self, *args_conv2d, **kwargs_conv2d):
        super(MaskConv2d, self).__init__(*args_conv2d, **kwargs_conv2d)


    def _out_dim(self, arr, dim=0):
        p = self.padding[dim]
        d = self.dilation[dim]
        k = self.kernel_size[dim]
        s = self.stride[dim]
        return (arr + 2*p - d*(k-1)-1)//s + 1


    def forward(self, x_pad, lengths):

        x_pad = super(MaskConv2d, self).forward(x_pad)
        lengths = self._out_dim(lengths)

        return x_pad, lengths
