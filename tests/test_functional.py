import pytest
import librosa
import numpy as np
import torch

from torchaudio_contrib.functional import (stft, phase_vocoder, magphase, amplitude_to_db, db_to_amplitude,
                                           complex_norm, mu_law_encoding, mu_law_decoding, apply_filterbank
                                           )


xfail = pytest.mark.xfail


def _num_stft_bins(signal_len, fft_len, hop_length, pad):
    return (signal_len + 2 * pad - fft_len + hop_length) // hop_length


def _approx_all_equal(x, y, atol=1e-7):
    return torch.all(torch.lt(torch.abs(torch.add(x, -y)), atol))


def _all_equal(x, y):
    return torch.all(torch.eq(x, y))


@pytest.mark.parametrize('fft_length', [512])
@pytest.mark.parametrize('hop_length', [256])
@pytest.mark.parametrize('waveform', [
    (torch.randn(1, 100000)),
    (torch.randn(1, 2, 100000)),
    pytest.param(torch.randn(1, 100), marks=xfail(raises=RuntimeError)),
])
@pytest.mark.parametrize('pad_mode', [
    # 'constant',
    'reflect',
])
def test_stft(waveform, fft_length, hop_length, pad_mode):
    """
    Test STFT for multi-channel signals.

    Padding: Value in having padding outside of torch.stft?
    """
    pad = fft_length // 2
    window = torch.hann_window(fft_length)
    complex_spec = stft(waveform, fft_length=fft_length, hop_length=hop_length, window=window, pad_mode=pad_mode)
    mag_spec, phase_spec = magphase(complex_spec)

    # == Test shape
    expected_size = list(waveform.size()[:-1])
    expected_size += [fft_length // 2 + 1, _num_stft_bins(
        waveform.size(-1), fft_length, hop_length, pad), 2]
    assert complex_spec.dim() == waveform.dim() + 2
    assert complex_spec.size() == torch.Size(expected_size)

    # == Test values
    fft_config = dict(n_fft=fft_length, hop_length=hop_length, pad_mode=pad_mode)
    # note that librosa *automatically* pad with fft_length // 2.
    expected_complex_spec = np.apply_along_axis(librosa.stft, -1,
                                                waveform.numpy(), **fft_config)
    expected_mag_spec, _ = librosa.magphase(expected_complex_spec)
    # Convert torch to np.complex
    complex_spec = complex_spec.numpy()
    complex_spec = complex_spec[..., 0] + 1j * complex_spec[..., 1]

    assert np.allclose(complex_spec, expected_complex_spec, atol=1e-5)
    assert np.allclose(mag_spec.numpy(), expected_mag_spec, atol=1e-5)


@pytest.mark.parametrize('rate', [0.5, 1.01, 1.3])
@pytest.mark.parametrize('complex_specgrams', [
    torch.randn(1, 2, 1025, 400, 2),
    torch.randn(1, 1025, 400, 2)
])
@pytest.mark.parametrize('hop_length', [256])
def test_phase_vocoder(complex_specgrams, rate, hop_length):

    class use_double_precision:
        def __enter__(self):
            self.default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float64)

        def __exit__(self, type, value, traceback):
            torch.set_default_dtype(self.default_dtype)

    # Due to cummulative sum, numerical error in using torch.float32 will
    # result in bottom right values of the stretched sectrogram to not
    # match with librosa
    with use_double_precision():

        complex_specgrams = complex_specgrams.type(torch.get_default_dtype())

        phase_advance = torch.linspace(0, np.pi * hop_length, complex_specgrams.shape[-3])[..., None]
        complex_specgrams_stretch = phase_vocoder(complex_specgrams, rate=rate, phase_advance=phase_advance)

        # == Test shape
        expected_size = list(complex_specgrams.size())
        expected_size[-2] = int(np.ceil(expected_size[-2] / rate))

        assert complex_specgrams.dim() == complex_specgrams_stretch.dim()
        assert complex_specgrams_stretch.size() == torch.Size(expected_size)

        # == Test values
        index = [0] * (complex_specgrams.dim() - 3) + [slice(None)] * 3
        mono_complex_specgram = complex_specgrams[index].numpy()
        mono_complex_specgram = mono_complex_specgram[..., 0] + \
            mono_complex_specgram[..., 1] * 1j
        expected_complex_stretch = librosa.phase_vocoder(
            mono_complex_specgram,
            rate=rate,
            hop_length=hop_length)

        complex_stretch = complex_specgrams_stretch[index].numpy()
        complex_stretch = complex_stretch[..., 0] + \
            1j * complex_stretch[..., 1]
        assert np.allclose(complex_stretch,
                           expected_complex_stretch, atol=1e-5)


@pytest.mark.parametrize('complex_tensor', [
    torch.randn(1, 2, 1025, 400, 2),
    torch.randn(1025, 400, 2)
])
@pytest.mark.parametrize('power', [1, 2, 0.7])
def test_complex_norm(complex_tensor, power):
    expected_norm_tensor = complex_tensor.pow(2).sum(-1).pow(power / 2)
    norm_tensor = complex_norm(complex_tensor, power)

    assert _approx_all_equal(expected_norm_tensor, norm_tensor, atol=1e-5)


@pytest.mark.parametrize('new_len', [120, 36])
@pytest.mark.parametrize('mag_spec', [
    torch.randn(1, 257, 391),
    torch.randn(1, 2, 257, 391),
])
def test_apply_filterbank(mag_spec, new_len):
    filterbank = torch.randn(mag_spec.size(-2), new_len)
    mag_spec_filterbanked = apply_filterbank(mag_spec, filterbank)
    assert mag_spec.size(-1) == mag_spec_filterbanked.size(-1)
    assert mag_spec_filterbanked.size(-2) == new_len
    assert mag_spec.dim() == mag_spec_filterbanked.dim()


@pytest.mark.parametrize('amplitude,db', [
    (torch.Tensor([0.000001, 0.0001, 0.1, 1.0, 10.0, 1000000.0]),
     torch.Tensor([-60.0, -40.0, -10.0, 0.0, 10.0, 60.0]))
])
def test_amplitude_db(amplitude, db):
    """Test amplitude_to_db and db_to_amplitude."""
    amplitude = np.sqrt(amplitude)
    assert _approx_all_equal(db, amplitude_to_db(amplitude, ref=1.0))
    assert _approx_all_equal(amplitude, db_to_amplitude(db, ref=1.0))
    # both ways
    assert _approx_all_equal(db_to_amplitude(amplitude_to_db(amplitude, ref=1.0), ref=1.0),
                             amplitude)
    assert _approx_all_equal(amplitude_to_db(db_to_amplitude(db, ref=1.0), ref=1.0),
                             db,
                             atol=1e-5)


@pytest.mark.parametrize('waveform', [
    torch.randn(1, 100000),
    (torch.randn(1, 2, 100000)),
])
@pytest.mark.parametrize('n_quantize', [256])
def test_mu_law(waveform, n_quantize):
    """test mu-law encoding and decoding"""

    def _test_mu_encoding(waveform, n_quantize):

        waveform = 2 * (waveform - 0.5)
        # manual computation
        mu = torch.tensor(n_quantize - 1, dtype=waveform.dtype)
        waveform_mu = waveform.sign() * torch.log1p(mu * waveform.abs()) / torch.log1p(mu)
        waveform_mu = ((waveform_mu + 1) / 2 * mu + 0.5).long()

        assert _all_equal(mu_law_encoding(waveform, n_quantize),
                          waveform_mu)

    def _test_mu_decoding(waveform, n_quantize):

        waveform_mu = torch.randint(low=0, high=n_quantize - 1,
                                    size=(1, 1024))

        # manual computation
        waveform_mu = waveform_mu.float()
        mu = torch.tensor(n_quantize - 1, dtype=waveform_mu.dtype)  # confused about dtype here..

        waveform = (waveform_mu / mu) * 2 - 1.
        waveform = waveform.sign() * (torch.exp(waveform.abs() * torch.log1p(mu)) - 1.) / mu

        assert _all_equal(mu_law_decoding(waveform_mu, n_quantize),
                          waveform)

    def _test_both_ways(waveform, n_quantize):
        waveform_mu = torch.randint(low=0, high=n_quantize - 1,
                                    size=(1, 1024))
        assert _all_equal(waveform_mu,
                          mu_law_encoding(mu_law_decoding(waveform_mu, n_quantize), n_quantize))

    _test_mu_encoding(waveform, n_quantize)
    _test_mu_decoding(waveform, n_quantize)
    _test_both_ways(waveform, n_quantize)
