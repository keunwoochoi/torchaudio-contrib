import torch
import numpy as np


def _get_time_values(sig_length, sr, hop):
    """
    Get the time axis values given the signal length, sample
    rate and hop size.
    """
    return torch.linspace(0, sig_length/sr, sig_length//hop+1)


def _get_freq_values(n_fft, sr):
    """
    Get the frequency axis values given the number of FFT bins
    and sample rate.
    """
    return torch.linspace(0, sr/2, n_fft//2 + 1)


def get_spectrogram_axis(sig_length, sr, n_fft=2048, hop=512):
    t = _get_time_values(sig_length, sr, hop)
    f = _get_freq_values(n_fft, sr)
    return t, f


def STFT(sig, n_fft=2048, hop=None, window=None, **kwargs):
    """A wrapper for torch.stft with some preset parameters.
    Returned value keeps both real and imageinary parts.
    For STFT magnitude, see spectrogram

    input -> (batch, channel, time) or (channel, time)
    output -> (batch, channel, freq, hop, complex) or (channel, freq, hop, complex)

    """
    if hop is None:
        hop = n_fft // 4
    if window is None:
        window = torch.hann_window(n_fft)

    if sig.dim() == 3:
        batch, channel, time = sig.size()
        out_shape = [batch, channel, n_fft//2+1, -1, 2]
    elif sig.dim() == 2:
        channel, time = sig.size()
        out_shape = [channel, n_fft//2+1, -1, 2]
    else:
        raise ValueError('Input tensor dim() must be either 2 or 3.')

    sig = sig.reshape(-1, time)
    stft = torch.stft(sig, n_fft, hop, window=window, **kwargs)
    stft = stft.reshape(out_shape)

    return stft


def spectrogram(sig, n_fft=2048, hop=None, window=None, power=1.0, **kwargs):
    """
    returns magnitude of spectrogram

    input -> (batch, channel, time) or (channel, time)
    output -> (batch, channel, freq, hop) or (channel, freq, hop)
    """
    stft = STFT(sig, n_fft, hop=hop, window=window, **kwargs)
    if power is None:
        return stft
    return stft.pow(2).sum(-1).pow(power / 2.0)


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


def melspectrogram(sig, n_mels=128, sr=44100, n_fft=2048, hop=None, window=None, **kwargs):
    """
    returns Melspectrogram
    """
    spec_amp = spectrogram(sig, n_fft, hop, window, power=1)
    mel_fb, _ = create_mel_filter(spec_amp.size(-1), sr, n_mels, **kwargs)
    mel_spec_amp = torch.matmul(spec_amp, mel_fb)
    return mel_spec_amp


def create_mel_filter(n_stft, sr, n_mels=128, f_min=0.0, f_max=None):
    '''
    Creates filter matrix to transform fft frequency bins into mel frequency bins.
    Equivalent to librosa.filters.mel(sr, n_fft, htk=True, norm=None).

    Output Tensor shape -> (n_mels, n_stft)
    '''
    # Convert to find mel lower/upper bounds
    f_max = f_max if f_max else sr // 2
    m_min = 0. if f_min == 0 else _hertz_to_mel(f_min)
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

    return fb, f_pts[:-2]


def amplitude_to_db(spec, ref=1.0, amin=1e-10, top_db=80.0):
    """
    Amplitude spectrogram to the db scale

    Input Tensor shape -> (freq, time)
    Output Tensor shape -> (freq, time)
    """
    power = spec**2
    return power_to_db(power, ref, amin, top_db)


def power_to_db(spec, ref=1.0, amin=1e-10, top_db=80.0):
    """
    Power spectrogram to the db scale

    Input Tensor shape -> (freq, time)
    Output Tensor shape -> (freq, time)
    """
    if amin <= 0:
        raise ParameterError('amin must be strictly positive')

    if callable(ref):
        ref_value = ref(spec_norm)
    else:
        ref_value = torch.tensor(ref)

    log_spec = 10*torch.log10(torch.clamp(spec, min=amin))
    log_spec -= 10*torch.log10(torch.clamp(ref_value, min=amin))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError('top_db must be non-negative')
        log_spec = torch.clamp(log_spec, min=(log_spec.max() - top_db))

    return log_spec


def db_to_power(spec_db, ref=1.0):
    """
    db-scale spectrogram to power spectrogram

    Input Tensor shape -> (freq, time)
    Output Tensor shape -> (freq, time)
    """
    return ref * torch.pow(10., spec_db * 0.1)


def phase_vocoder(spec, rate, hop, n_fft):
    """
    Phase vocoder. Given a STFT tensor, speed up in time 
    without modifying pitch by a factor of `rate`.

    Input Tensor shape -> (batch, channel, freq, time, 2)
    Output Tensor shape -> (batch, channel, freq, time//rate+1, 2)
    """

    fft_size = n_fft//2 + 1
    time_steps = torch.arange(0, spec.size(
        3), rate, device=spec.device)  # (new_time)
    phi_advance = torch.linspace(
        0, np.pi * hop, fft_size, device=spec.device)[..., None]

    alphas = (time_steps % 1)  # .unsqueeze(1) # (new_time)

    phase_0 = torch.atan2(spec[:, :, :, :1, 1], spec[:, :, :, :1, 0])

    # Time Padding
    pad_shape = [0, 0]+[0, 2]+[0]*6
    spec = torch.nn.functional.pad(spec, pad_shape)

    spec_0 = spec[:, :, :, time_steps.long()]  # (new_time, freq, 2)
    spec_1 = spec[:, :, :, (time_steps + 1).long()]  # (new_time, freq, 2)

    spec_0_angle = torch.atan2(
        spec_0[..., 1], spec_0[..., 0])  # (new_time, freq)
    spec_1_angle = torch.atan2(
        spec_1[..., 1], spec_1[..., 0])  # (new_time, freq)

    spec_0_norm = torch.norm(spec_0, dim=-1)  # (new_time, freq)
    spec_1_norm = torch.norm(spec_1, dim=-1)  # (new_time, freq)

    spec_phase = spec_1_angle - spec_0_angle - phi_advance  # (new_time, freq)
    spec_phase = spec_phase - 2 * np.pi * \
        torch.round(spec_phase / (2 * np.pi))  # (new_time, freq)

    # Compute Phase Accum
    phase = spec_phase + phi_advance  # (new_time, freq)

    phase = torch.cat([phase_0, phase[:, :, :, :-1]], dim=-1)

    phase_acc = torch.cumsum(phase, -1)  # (new_time, freq)

    mag = alphas * spec_1_norm + (1-alphas) * spec_0_norm  # (new_time, freq)

    spec_stretch_real = mag * torch.cos(phase_acc)  # (new_time, freq)
    spec_stretch_imag = mag * torch.sin(phase_acc)  # (new_time, freq)

    spec_stretch = torch.stack([spec_stretch_real, spec_stretch_imag], dim=-1)

    return spec_stretch
