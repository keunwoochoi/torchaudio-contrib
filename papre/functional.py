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
    return torch.linspace(0, sr/2, n_fft//2 +1)

def get_spectrogram_axis(sig_length, sr, n_fft=2048, hop=512):
    t = _get_time_values(sig_length, sr, hop)
    f = _get_freq_values(n_fft, sr)
    return t, f


def STFT(sig, n_fft=2048, hop=None, window=None, **kwargs):
    """A wrapper for torch.stft with some preset parameters.
    Returned value keeps both real and imageinary parts.
    For STFT magnitude, see spectrogram

    """
    if hop is None:
        hop = n_fft // 4
    if window is None:
        window = torch.hann_window(n_fft)
    return torch.stft(sig, n_fft, hop, window=window, **kwargs).transpose(1,2)


def spectrogram(sig, n_fft=2048, hop=None, window=None, power=1.0, **kwargs):
    """
    returns magnitude of spectrogram
    """
    stft = STFT(sig, n_fft, hop=hop, window=window, **kwargs)
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



def melspectrogram(sig, sr, n_fft=2048, hop=None, window=None, power=1.0, **kwargs):
    """
    returns Melspectrogram
    """
    spec_amp = spectrogram(sig, n_fft, hop, window, power=power, **kwargs)
    mel_fb, _ = create_mel_filter(spec_amp.size(-1), sr, **kwargs)
    mel_spec_amp = torch.matmul(spec_amp, mel_fb)
    return mel_spec_amp


def create_mel_filter(n_stft, sr, n_mels=128, f_min=0.0, f_max=None):
    '''
    Creates filter matrix to transform fft frequency bins into mel frequency bins.
    Equivalent to librosa.filters.mel(sr, n_fft, htk=True, norm=None).
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
    slopes = f_pts.unsqueeze(0) - stft_freqs.unsqueeze(1)  # (n_stft, n_mels + 2)
    
    down_slopes = (-1. * slopes[:, :-2]) / f_diff[:-1]  # (n_stft, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_stft, n_mels)
    fb = torch.clamp(torch.min(down_slopes, up_slopes), min=0.)

    return fb, f_pts[:-2]


def amplitude_to_db(spec, ref=1.0, amin=1e-10, top_db=80.0):
    """
    Amplitude spectrogram to the db scale
    """
    power = spec**2
    return power_to_db(power, ref, amin, top_db)


def power_to_db(spec, ref=1.0, amin=1e-10, top_db=80.0):
    """
    Power spectrogram to the db scale

    spec -> (time, freq, complex) or (time, freq)
    """
    if amin <= 0:
        raise ParameterError('amin must be strictly positive')

    if spec.size(-1) == 2: # obv make this better
        spec_norm = torch.norm(spec, p=2, dim=-1)
    else:
        spec_norm = spec

    if callable(ref):
        ref_value = ref(spec_norm)
    else:
        ref_value = torch.tensor(ref)

    log_spec = 10*torch.log10( torch.clamp(spec_norm, min=amin) )
    log_spec -= 10*torch.log10( max(ref_value, torch.tensor(amin)) )
    
    if top_db is not None:
        if top_db < 0:
            raise ParameterError('top_db must be non-negative')
        log_spec = torch.clamp(log_spec, min=(log_spec.max() - top_db))
    
    return log_spec
    

def db_to_power(spec_db, ref=1.0):
    """
    """
    
    return ref * torch.pow(10., spec_db * 0.1) 


def pseudo_cqt():
    """computing pseudo-cqt
    """
    pass