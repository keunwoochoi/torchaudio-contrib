import torch
import math
import torch.nn.functional as F


def stft(signal, fft_len, hop_len, window,
         pad=0, pad_mode="reflect", **kwargs):
    """
    Wrap torch.stft allowing for multi-channel stft.

    Args:
        signal (Tensor): Tensor of audio of size (channel, time)
            or (batch, channel, time).
        fft_len (int): FFT window size.
        hop_len (int): Number audio of frames between STFT columns.
        window (Tensor): 1-D tensor.
        pad (int): Amount of padding to apply to signal.
        pad_mode: padding method (see torch.nn.functional.pad).
        **kwargs: Other torch.stft parameters, see torch.stft for more details.

    Returns:
        Tensor: (batch, channel, num_bins, time, complex)
            or (channel, num_bins, time, complex)

    Example:
        >>> signal = torch.randn(16, 2, 10000)
        >>> # window_length <= fft_len
        >>> window = torch.hamming_window(window_length=2048)
        >>> x = stft(signal, 2048, 512, window)
        >>> x.shape
        torch.Size([16, 2, 1025, 20])
    """

    # (!) Only 3D, 4D, 5D padding with non-constant
    # padding are supported for now.
    if pad > 0:
        signal = F.pad(signal, (pad, pad), pad_mode)

    leading_dims = signal.shape[:-1]

    signal = signal.reshape(-1, signal.size(-1))

    spect = torch.stft(signal, fft_len, hop_len, window=window,
                       win_length=window.size(0), **kwargs)
    spect = spect.reshape(leading_dims + spect.shape[1:])

    return spect


def complex_norm(tensor, power=1.0):
    """
    Normalize complex input.
    """
    if power == 1.:
        return torch.norm(tensor, 2, -1)
    return torch.norm(tensor, 2, -1).pow(power)


def create_mel_filter(num_bands, sample_rate, min_freq,
                      max_freq, num_bins, to_hertz, from_hertz):
    """
    Creates filter matrix to transform fft frequency bins
    into mel frequency bins.
    Equivalent to librosa.filters.mel(sample_rate, fft_len, htk=True, norm=None).

    Args:
        num_bands (int): number of mel bins.
        sample_rate (int): sample rate of audio signal.
        min_freq (float): minimum frequency.
        max_freq (float): maximum frequency.
        num_bins (int): number of filter banks from stft.
        to_hertz (function): convert from mel freq to hertz
        from_hertz (function): convert from hertz to mel freq

    Returns:
        filterbank (Tensor): (num_bins, num_bands)
    """
    # Convert to find mel lower/upper bounds
    m_min = from_hertz(min_freq)
    m_max = from_hertz(max_freq)

    # Compute stft frequency values
    stft_freqs = torch.linspace(min_freq, max_freq, num_bins)

    # Find mel values, and convert them to frequency units
    m_pts = torch.linspace(m_min, m_max, num_bands + 2)
    f_pts = to_hertz(m_pts)
    f_diff = f_pts[1:] - f_pts[:-1]  # (num_bands + 1)

    # (num_bins, num_bands + 2)
    slopes = f_pts.unsqueeze(0) - stft_freqs.unsqueeze(1)

    down_slopes = (-1. * slopes[:, :-2]) / f_diff[:-1]  # (num_bins, num_bands)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (num_bins, num_bands)
    filterbank = torch.clamp(torch.min(down_slopes, up_slopes), min=0.)

    return filterbank


def apply_filterbank(spect, filterbank):
    """
    Transform spectrogram given a filterbank matrix.

    Args:
        spect (Tensor): (batch, channel, num_bins, time)
        filterbank (Tensor): (num_bins, num_bands)

    Returns:
        (Tensor): (batch, channel, num_bands, time)
    """
    return torch.matmul(spect.transpose(-2, -1), filterbank).transpose(-2, -1)


def angle(tensor):
    """
    Return angle of a complex tensor with shape (*, 2).
    """
    return torch.atan2(tensor[..., 1], tensor[..., 0])


def magphase(spect, power=1.):
    """
    Separate a complex-valued spectrogram with shape (*,2)
    into its magnitude and phase.
    """
    mag = complex_norm(spect, power)
    phase = angle(spect)
    return mag, phase


def phase_vocoder(spect, rate, phi_advance):
    """
    Phase vocoder. Given a STFT tensor, speed up in time
    without modifying pitch by a factor of `rate`.

    Args:
        spect (Tensor): (batch, channel, num_bins, time, 2)
        rate (float): Speed-up factor
        phi_advance (Tensor): Expected phase advance in each bin. (num_bins, 1)

    Returns:
      (Tensor): (batch, channel, num_bins, new_bins, 2) with new_bins = num_bins//rate+1
    """

    time_steps = torch.arange(0, spect.size(
        3), rate, device=spect.device)  # (new_bins,)

    alphas = (time_steps % 1)  # (new_bins,)

    phase_0 = angle(spect[:, :, :, :1])

    # Time Padding
    pad_shape = [0, 0] + [0, 2] + [0] * 6
    spect = torch.nn.functional.pad(spect, pad_shape)

    spect_0 = spect[:, :, :, time_steps.long()]  # (new_bins, num_bins, 2)
    # (new_bins, num_bins, 2)
    spect_1 = spect[:, :, :, (time_steps + 1).long()]

    spect_0_angle = angle(spect_0)  # (new_bins, num_bins)
    spect_1_angle = angle(spect_1)  # (new_bins, num_bins)

    spect_0_norm = torch.norm(spect_0, dim=-1)  # (new_bins, num_bins)
    spect_1_norm = torch.norm(spect_1, dim=-1)  # (new_bins, num_bins)

    spect_phase = spect_1_angle - spect_0_angle - \
        phi_advance  # (new_bins, num_bins)
    spect_phase = spect_phase - 2 * math.pi * \
        torch.round(spect_phase / (2 * math.pi))  # (new_bins, num_bins)

    # Compute Phase Accum
    phase = spect_phase + phi_advance  # (new_bins, num_bins)

    phase = torch.cat([phase_0, phase[:, :, :, :-1]], dim=-1)

    phase_acc = torch.cumsum(phase, -1)  # (new_bins, num_bins)

    mag = alphas * spect_1_norm + (1 - alphas) * \
        spec_0_norm  # (time//rate+1, num_bins)

    spect_stretch_real = mag * torch.cos(phase_acc)  # (new_bins, num_bins)
    spect_stretch_imag = mag * torch.sin(phase_acc)  # (new_bins, num_bins)

    spect_stretch = torch.stack(
        [spect_stretch_real, spect_stretch_imag], dim=-1)

    return spect_stretch
