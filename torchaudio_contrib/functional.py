import torch
import math
import torch.nn.functional as F


def stft_defaults(n_fft, hop_length, len_win, window):
    '''
    Should function outisde torchaudio_contrib.stft since 
    torchaudio_contrib.STFT will use it outisde of forward() (?).
    '''
    hop_length = n_fft // 4 if hop_length is None else hop_length

    if window is None:
        length = n_fft if len_win is None else len_win
        window = torch.hann_window(length)
        if not isinstance(window, torch.Tensor):
            raise TypeError('window must be a of type torch.Tensor')

    return n_fft, hop_length, window


def _stft(x, n_fft, hop_length, window, pad, pad_mode, **kwargs):
    """
    Wrap torch.stft allowing for multi-channel stft. 

    Args:
        x (Tensor): Tensor of audio of size (channel, signal) or (batch, channel, signal).
        n_fft (int): FFT window size.
        hop_length (int): Number audio of frames between STFT columns.
        len_win (int): Size of stft window.
        window (Tensor): 1-D tensor.
        pad (int): Amount of padding to apply to signal.
        pad_mode: padding method (see torch.nn.functional.pad).
        **kwargs: Other torch.stft parameters, see torch.stft for more details.

    Returns:
        Tensor: (batch, channel, freq, hop, complex) or (channel, freq, hop, complex)

    """

    # (!) Only 3D, 4D, 5D padding with non-constant
    # padding are supported for now.
    if pad > 0:
        x = F.pad(x, (pad, pad), pad_mode)

    if x.dim() == 3:
        batch, channel, time = x.size()
        out_shape = [batch, channel, n_fft//2+1, -1, 2]
    elif x.dim() == 2:
        channel, time = x.size()
        out_shape = [channel, n_fft//2+1, -1, 2]
    else:
        raise ValueError('Input tensor dim() must be either 2 or 3.')

    x = x.reshape(-1, time)

    stft_out = torch.stft(x, n_fft, hop_length, window=window,
                          win_length=window.size(0), **kwargs)
    stft_out = stft_out.reshape(out_shape)

    return stft_out


def stft(x, n_fft=2048, hop_length=None, len_win=None,
         window=None, pad=0, pad_mode="reflect", **kwargs):
    """
    Wraps _stft setting default values and correct window device. 
    See torchaudio_contrib.STFT for more details.
    """
    n_fft, hop_length, window = stft_defaults(
        n_fft, hop_length, len_win, window)
    return _stft(x, n_fft=n_fft, hop_length=hop_length,
                 window=window.to(x.device), pad=pad, pad_mode=pad_mode, **kwargs)


def complex_norm(x, power=1.0):
    """
    Normalize complex tensor.
    """
    return x.pow(2).sum(-1).pow(power / 2.0)


def spectrogram(x, n_fft=2048, hop_length=None, len_win=None,
                window=None, pad=0, pad_mode="reflect", power=1., **kwargs):
    """
    Compute the spectrogram of a given signal. 
    See torchaudio_contrib.Spectrogram for more details. 
    """
    return complex_norm(stft(x, n_fft, hop_length, len_win,
                             window, pad, pad_mode, **kwargs), power=power)


def _hertz_to_mel(f):
    '''
    Converting frequency into mel values using HTK formula
    '''
    return 2595. * torch.log10(torch.tensor(1.) + (f / 700.))


def _mel_to_hertz(mel):
    '''
    Converting mel values into frequency using HTK formula
    '''
    return 700. * (10**(mel / 2595.) - 1.)


def create_mel_filter(n_mels, sr, f_min, f_max, n_stft):
    '''
    Creates filter matrix to transform fft frequency bins into mel frequency bins.
    Equivalent to librosa.filters.mel(sr, n_fft, htk=True, norm=None).

    Args:
        n_mels (int): number of mel bins.
        sr (int): sample rate of audio signal.
        f_max (float, optional): maximum frequency.
        f_min (float): minimum frequency.
        n_stft (int, optional): number of filter banks from stft.

    Returns:
        fb (Tensor): (n_stft, n_mels)
    '''
    # Convert to find mel lower/upper bounds
    f_max = f_max if f_max else sr // 2
    n_stft = n_stft if n_stft else 1025

    m_min = _hertz_to_mel(f_min)
    m_max = _hertz_to_mel(f_max)

    # Compute stft frequency values
    stft_freqs = torch.linspace(f_min, f_max, n_stft)

    # Find mel values, and convert them to frequency units
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hertz(m_pts)

    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    # (n_stft, n_mels + 2)
    slopes = f_pts.unsqueeze(0) - stft_freqs.unsqueeze(1)

    down_slopes = (-1. * slopes[:, :-2]) / f_diff[:-1]  # (n_stft, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_stft, n_mels)
    fb = torch.clamp(torch.min(down_slopes, up_slopes), min=0.)

    return fb


def melspectrogram(x, n_mels=128, sr=44100, f_min=0.0, f_max=None, **kwargs):
    """
    Compute the melspectrogram of a given signal. 
    See torchaudio_contrib.Melspectrogram for more details.
    """
    spec = spectrogram(x, **kwargs)
    fb = create_mel_filter(
        n_mels=n_mels,
        sr=sr,
        f_min=f_min,
        f_max=f_max,
        n_stft=spec.size(-2))
    fb = fb.to(x.device)

    return torch.matmul(spec.transpose(-2, -1), fb).transpose(-2, -1)


def angle(tensor):
    """ 
    Return angle of a complex tensor with shape (*, 2).
    """
    return torch.atan2(tensor[..., 1], tensor[..., 0])


def magphase(spec, power=1.):
    """
    Separate a complex-valued spectrogram with shape (*,2)
    into its magnitude and phase.
    """
    mag = spec.pow(2).sum(-1).pow(power/2)
    phase = angle(spec)
    return mag, phase


def phase_vocoder(spec, rate, phi_advance):
    """
    Phase vocoder. Given a STFT tensor, speed up in time 
    without modifying pitch by a factor of `rate`.


    Input Tensor shape -> (batch, channel, freq, time, 2)
    Output Tensor shape -> (batch, channel, freq, time//rate+1, 2)
    """

    time_steps = torch.arange(0, spec.size(
        3), rate, device=spec.device)  # (new_time)

    alphas = (time_steps % 1)  # (new_time)

    phase_0 = angle(spec[:, :, :, :1])

    # Time Padding
    pad_shape = [0, 0]+[0, 2]+[0]*6
    spec = torch.nn.functional.pad(spec, pad_shape)

    spec_0 = spec[:, :, :, time_steps.long()]  # (new_time, freq, 2)
    spec_1 = spec[:, :, :, (time_steps + 1).long()]  # (new_time, freq, 2)

    spec_0_angle = angle(spec_0)  # (new_time, freq)
    spec_1_angle = angle(spec_1)  # (new_time, freq)

    spec_0_norm = torch.norm(spec_0, dim=-1)  # (new_time, freq)
    spec_1_norm = torch.norm(spec_1, dim=-1)  # (new_time, freq)

    spec_phase = spec_1_angle - spec_0_angle - phi_advance  # (new_time, freq)
    spec_phase = spec_phase - 2 * math.pi * \
        torch.round(spec_phase / (2 * math.pi))  # (new_time, freq)

    # Compute Phase Accum
    phase = spec_phase + phi_advance  # (new_time, freq)

    phase = torch.cat([phase_0, phase[:, :, :, :-1]], dim=-1)

    phase_acc = torch.cumsum(phase, -1)  # (new_time, freq)

    mag = alphas * spec_1_norm + (1-alphas) * spec_0_norm  # (new_time, freq)

    spec_stretch_real = mag * torch.cos(phase_acc)  # (new_time, freq)
    spec_stretch_imag = mag * torch.sin(phase_acc)  # (new_time, freq)

    spec_stretch = torch.stack([spec_stretch_real, spec_stretch_imag], dim=-1)

    return spec_stretch
