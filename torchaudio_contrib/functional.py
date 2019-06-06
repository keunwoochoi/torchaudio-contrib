import torch
import math
import torch.nn.functional as F


def _mel_to_hertz(mel, htk):
    """
    Converting mel values into frequency
    """
    mel = torch.as_tensor(mel).type(torch.get_default_dtype())

    if htk:
        return 700. * (10 ** (mel / 2595.) - 1.)

    f_min = 0.0
    f_sp = 200.0 / 3
    hz = f_min + f_sp * mel

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    return torch.where(mel >= min_log_mel, min_log_hz *
                       torch.exp(logstep * (mel - min_log_mel)), hz)


def _hertz_to_mel(hz, htk):
    """
    Converting frequency into mel values
    """
    hz = torch.as_tensor(hz).type(torch.get_default_dtype())

    if htk:
        return 2595. * torch.log10(torch.tensor(1., dtype=torch.get_default_dtype()) + (hz / 700.))

    f_min = 0.0
    f_sp = 200.0 / 3

    mel = (hz - f_min) / f_sp

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    return torch.where(hz >= min_log_hz, min_log_mel +
                       torch.log(hz / min_log_hz) / logstep, mel)


def stft(waveforms, fft_length, hop_length=None, win_length=None, window=None,
         center=True, pad_mode='reflect', normalized=False, onesided=True):
    """Compute a short-time Fourier transform of the input waveform(s).
    It wraps `torch.stft` but after reshaping the input audio
    to allow for `waveforms` that `.dim()` >= 3.
    It follows most of the `torch.stft` default value, but for `window`,
    if it's not specified (`None`), it uses hann window.

    Args:
        waveforms (Tensor): Tensor of audio signal of size `(*, channel, time)`
        fft_length (int): FFT size [sample]
        hop_length (int): Hop size [sample] between STFT frames.
            Defaults to `fft_length // 4` (75%-overlapping windows) by `torch.stft`.
        win_length (int): Size of STFT window.
            Defaults to `fft_length` by `torch.stft`.
        window (Tensor): 1-D Tensor.
            Defaults to Hann Window of size `win_length` *unlike* `torch.stft`.
        center (bool): Whether to pad `waveforms` on both sides so that the
            `t`-th frame is centered at time `t * hop_length`.
            Defaults to `True` by `torch.stft`.
        pad_mode (str): padding method (see `torch.nn.functional.pad`).
            Defaults to `'reflect'` by `torch.stft`.
        normalized (bool): Whether the results are normalized.
            Defaults to `False` by `torch.stft`.
        onesided (bool): Whether the half + 1 frequency bins are returned to remove
            the symmetric part of STFT of real-valued signal.
            Defaults to `True` by `torch.stft`.

    Returns:
        complex_specgrams (Tensor): `(*, channel, num_freqs, time, complex=2)`

    Example:
        >>> waveforms = torch.randn(16, 2, 10000)  # (batch, channel, time)
        >>> x = stft(waveforms, 2048, 512)
        >>> x.shape
        torch.Size([16, 2, 1025, 20])
    """
    leading_dims = waveforms.shape[:-1]

    waveforms = waveforms.reshape(-1, waveforms.size(-1))

    if window is None:
        if win_length is None:
            window = torch.hann_window(fft_length)
        else:
            window = torch.hann_window(win_length)

    complex_specgrams = torch.stft(waveforms, n_fft=fft_length, hop_length=hop_length,
                                   win_length=win_length, window=window,
                                   center=center, pad_mode=pad_mode, normalized=normalized,
                                   onesided=onesided)
    complex_specgrams = complex_specgrams.reshape(leading_dims + complex_specgrams.shape[1:])

    return complex_specgrams


def complex_norm(complex_tensor, power=1.0):
    """Compute the norm of complex tensor input

    Args:
        complex_tensor (Tensor): Tensor shape of `(*, complex=2)`
        power (float): Power of the norm. Defaults to `1.0`.

    Returns:
        Tensor: norm of the input tensor, shape of `(*, )`
    """
    if power == 1.0:
        return torch.norm(complex_tensor, 2, -1)
    return torch.norm(complex_tensor, 2, -1).pow(power)


def create_mel_filter(num_freqs, num_mels, min_freq, max_freq, htk):
    """
    Creates filter matrix to transform fft frequency bins
    into mel frequency bins.
    Equivalent to librosa.filters.mel(sample_rate, fft_len, htk=True, norm=None).

    Args:
        num_freqs (int): number of filter banks from stft.
        num_mels (int): number of mel bins.
        min_freq (float): minimum frequency.
        max_freq (float): maximum frequency.
        htk (bool): whether following htk-mel scale or not

    Returns:
        mel_filterbank (Tensor): (num_freqs, num_mels)
    """
    # Convert to find mel lower/upper bounds
    m_min = _hertz_to_mel(min_freq, htk)
    m_max = _hertz_to_mel(max_freq, htk)

    # Compute stft frequency values
    stft_freqs = torch.linspace(min_freq, max_freq, num_freqs)

    # Find mel values, and convert them to frequency units
    m_pts = torch.linspace(m_min, m_max, num_mels + 2)
    f_pts = _mel_to_hertz(m_pts, htk)
    f_diff = f_pts[1:] - f_pts[:-1]  # (num_mels + 1)

    # (num_freqs, num_mels + 2)
    slopes = f_pts.unsqueeze(0) - stft_freqs.unsqueeze(1)

    down_slopes = (-1. * slopes[:, :-2]) / f_diff[:-1]  # (num_freqs, num_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (num_freqs, num_mels)
    mel_filterbank = torch.clamp(torch.min(down_slopes, up_slopes), min=0.)

    return mel_filterbank


def apply_filterbank(mag_specgrams, filterbank):
    """
    Transform spectrogram given a filterbank matrix.

    Args:
        mag_specgrams (Tensor): (batch, channel, num_freqs, time)
        filterbank (Tensor): (num_freqs, num_bands)

    Returns:
        (Tensor): (batch, channel, num_bands, time)
    """
    return torch.matmul(mag_specgrams.transpose(-2, -1), filterbank).transpose(-2, -1)


def angle(complex_tensor):
    """
    Return angle of a complex tensor with shape (*, 2).
    """
    return torch.atan2(complex_tensor[..., 1], complex_tensor[..., 0])


def magphase(complex_tensor, power=1.):
    """
    Separate a complex-valued spectrogram with shape (*,2)
    into its magnitude and phase.
    """
    mag = complex_norm(complex_tensor, power)
    phase = angle(complex_tensor)
    return mag, phase


def phase_vocoder(spect, rate, phi_advance):
    """
    Phase vocoder. Given a STFT tensor, speed up in time
    without modifying pitch by a factor of `rate`.

    Args:
        spect (Tensor): (batch, channel, num_bins, time, complex=2)
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
          spect_0_norm  # (time//rate+1, num_bins)

    spect_stretch_real = mag * torch.cos(phase_acc)  # (new_bins, num_bins)
    spect_stretch_imag = mag * torch.sin(phase_acc)  # (new_bins, num_bins)

    spect_stretch = torch.stack(
        [spect_stretch_real, spect_stretch_imag], dim=-1)

    return spect_stretch


def amplitude_to_db(x, ref=1.0, amin=1e-7):
    """
    Amplitude-to-decibel conversion (logarithmic mapping with base=10)
    By using `amin=1e-7`, it assumes 32-bit floating point input. If the
    data precision differs, use approproate `amin` accordingly.

    Args:
        x (Tensor): Input amplitude
        ref (float): Amplitude value that is equivalent to 0 decibel
        amin (float): Minimum amplitude. Any input that is smaller than `amin` is
            clamped to `amin`.
    Returns:
        (Tensor): same size of x, after conversion
    """
    x = torch.clamp(x, min=amin)
    return 10.0 * (torch.log10(x) - torch.log10(torch.tensor(ref,
                                                             device=x.device,
                                                             requires_grad=False,
                                                             dtype=x.dtype)))


def db_to_amplitude(x, ref=1.0):
    """
    Decibel-to-amplitude conversion (exponential mapping with base=10)

    Args:
        x (Tensor): Input in decibel to be converted
        ref (float): Amplitude value that is equivalent to 0 decibel

    Returns:
        (Tensor): same size of x, after conversion
    """
    return torch.pow(10.0, x / 10.0 + torch.log10(torch.tensor(ref,
                                                               device=x.device,
                                                               requires_grad=False,
                                                               dtype=x.dtype)))


def mu_law_encoding(x, n_quantize=256):
    """Apply mu-law encoding to the input tensor.
    Usually applied to waveforms

    Args:
        x (Tensor): input value
        n_quantize (int): quantization level. For 8-bit encoding, set 256 (2 ** 8).

    Returns:
        (Tensor): same size of x, after encoding

    """
    if not x.dtype.is_floating_point:
        x = x.to(torch.float)
    mu = torch.tensor(n_quantize - 1, dtype=x.dtype, requires_grad=False)  # confused about dtype here..

    x_mu = x.sign() * torch.log1p(mu * x.abs()) / torch.log1p(mu)
    x_mu = ((x_mu + 1) / 2 * mu + 0.5).long()
    return x_mu


def mu_law_decoding(x_mu, n_quantize=256, dtype=torch.get_default_dtype()):
    """Apply mu-law decoding (expansion) to the input tensor.

    Args:
        x_mu (Tensor): mu-law encoded input
        n_quantize (int): quantization level. For 8-bit decoding, set 256 (2 ** 8).
        dtype: specifies `dtype` for the decoded value. Default: `torch.get_default_dtype()`

    Returns:
        (Tensor): mu-law decoded tensor
    """
    if not x_mu.dtype.is_floating_point:
        x_mu = x_mu.to(dtype)
    mu = torch.tensor(n_quantize - 1, dtype=x_mu.dtype, requires_grad=False)  # confused about dtype here..
    x = (x_mu / mu) * 2 - 1.
    x = x.sign() * (torch.exp(x.abs() * torch.log1p(mu)) - 1.) / mu
    return x
