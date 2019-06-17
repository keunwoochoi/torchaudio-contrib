"""
Test the layers. Currently only on cpu since travis doesn't have GPU.
"""
import unittest
import pytest
import librosa
import numpy as np
import torch
import torch.nn as nn
from torchaudio_contrib.layers import (STFT, Spectrogram, MelFilterbank, AmplitudeToDb, TimeStretch,
                                       ComplexNorm, ApplyFilterbank)

from test_functional import _num_stft_bins

xfail = pytest.mark.xfail


@pytest.mark.parametrize('fft_len', [512])
@pytest.mark.parametrize('hop_length', [256])
@pytest.mark.parametrize('waveform', [
    torch.randn(1, 100000)
])
@pytest.mark.parametrize('pad_mode', [
    # 'constant',
    'reflect',
])
def test_STFT(waveform, fft_len, hop_length, pad_mode):
    """
    Test STFT for multi-channel signals.

    Padding: Value in having padding outside of torch.stft?
    """
    pad = fft_len // 2
    layer = STFT(fft_length=fft_len, hop_length=hop_length, pad_mode=pad_mode)

    assert torch.is_tensor(layer.window)
    assert not layer.window.requires_grad
    assert layer.window.size(0) <= layer.fft_length


@pytest.mark.parametrize('rate', [0.7])
@pytest.mark.parametrize('complex_specgrams', [
    torch.randn(1, 2, 1025, 400, 2),
    torch.randn(1, 1025, 400, 2)
])
@pytest.mark.parametrize('hop_length', [256])
def test_TimeStretch(complex_specgrams, rate, hop_length):

    layer = TimeStretch(hop_length=hop_length, num_freqs=complex_specgrams.shape[-3])

    assert torch.is_tensor(layer.phase_advance)
    assert not layer.phase_advance.requires_grad


@pytest.mark.parametrize('fft_length', [512])
@pytest.mark.parametrize('hop_length', [256])
@pytest.mark.parametrize('waveform', [
    torch.randn(1, 100000),
    torch.randn(1, 2, 100000)
])
@pytest.mark.parametrize('pad_mode', [
    # 'constant',
    'reflect',
])
def test_SpectrogramDb(waveform, fft_length, hop_length, pad_mode):

    ref, amin = 1.0, 1e-7
    window = torch.hann_window(fft_length)
    model = torch.nn.Sequential(*Spectrogram(fft_length, hop_length=hop_length, window=window, pad_mode=pad_mode),
                                AmplitudeToDb(ref=ref, amin=amin))
    db_spec = model(waveform).numpy()

    fft_config = dict(n_fft=fft_length, hop_length=hop_length, pad_mode=pad_mode)
    expected_db_spec = np.abs(np.apply_along_axis(librosa.stft, -1,
                              waveform.numpy(), **fft_config))

    db_config = dict(ref=ref, amin=amin, top_db=None)
    expected_db_spec = np.apply_along_axis(librosa.power_to_db,
                                           -1,
                                           expected_db_spec**2,
                                           **db_config)

    assert np.allclose(db_spec, expected_db_spec, atol=1e-2), np.abs(expected_db_spec - db_spec).max()


@pytest.mark.parametrize('fft_length', [512])
@pytest.mark.parametrize('num_mels', [128])
@pytest.mark.parametrize('hop_length', [256])
@pytest.mark.parametrize('waveform', [
    torch.randn(1, 2, 100000),
    torch.randn(4, 100000)
])
@pytest.mark.parametrize('rate', [0.7])
def test_MelspectrogramStretch(waveform, fft_length, num_mels, hop_length, rate):

    num_freqs = fft_length // 2 + 1
    fb = MelFilterbank(num_freqs=num_freqs, num_mels=num_mels, max_freq=1.0).get_filterbank()
    model = nn.Sequential(STFT(fft_length, hop_length=hop_length),
                          TimeStretch(hop_length=hop_length, num_freqs=num_freqs, fixed_rate=rate),
                          ComplexNorm(power=2.0),
                          ApplyFilterbank(fb))
    mel_spec = model(waveform)
    num_bins = _num_stft_bins(waveform.size(-1), fft_length, hop_length, fft_length // 2)

    assert mel_spec.size(-2) == num_mels
    assert mel_spec.size(-1) == np.ceil(num_bins / rate)

if __name__ == '__main__':
    unittest.main()
