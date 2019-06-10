"""
Test the layers. Currently only on cpu since travis doesn't have GPU.
"""
import unittest
import pytest
import librosa
import numpy as np
import torch
import torch.nn as nn
from torchaudio_contrib.layers import (
    STFT, ComplexNorm, ApplyFilterbank, Spectrogram, Melspectrogram,
    MelFilterbank, AmplitudeToDb, DbToAmplitude, MuLawEncoding, MuLawDecoding,
    TimeStretch
)
from torchaudio_contrib.functional import magphase


xfail = pytest.mark.xfail


def _num_stft_bins(signal_len, fft_len, hop_length, pad):
    return (signal_len + 2 * pad - fft_len + hop_length) // hop_length


def _seed(seed=1234):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def _approx_all_equal(x, y, atol=1e-7):
    return torch.all(torch.lt(torch.abs(torch.add(x, -y)), atol))


def _all_equal(x, y):
    return torch.all(torch.eq(x, y))


@pytest.mark.parametrize('fft_len', [512])
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
def test_STFT(waveform, fft_len, hop_length, pad_mode):
    """
    Test STFT for multi-channel signals.

    Padding: Value in having padding outside of torch.stft?
    """
    pad = fft_len // 2
    layer = STFT(fft_length=fft_len, hop_length=hop_length, pad_mode=pad_mode)
    complex_spec = layer(waveform)
    mag_spec, phase_spec = magphase(complex_spec)

    # == Test shape
    expected_size = list(waveform.size()[:-1])
    expected_size += [fft_len // 2 + 1, _num_stft_bins(
        waveform.size(-1), fft_len, hop_length, pad), 2]
    assert complex_spec.dim() == waveform.dim() + 2
    assert complex_spec.size() == torch.Size(expected_size)

    # == Test values
    fft_config = dict(n_fft=fft_len, hop_length=hop_length, pad_mode=pad_mode)
    # note that librosa *automatically* pad with fft_len // 2.
    expected_complex_spec = np.apply_along_axis(librosa.stft, -1,
                                                waveform.numpy(), **fft_config)
    expected_mag_spec, _ = librosa.magphase(expected_complex_spec)
    # Convert torch to np.complex
    complex_spec = complex_spec.numpy()
    complex_spec = complex_spec[..., 0] + 1j * complex_spec[..., 1]

    assert np.allclose(complex_spec, expected_complex_spec, atol=1e-5)
    assert np.allclose(mag_spec.numpy(), expected_mag_spec, atol=1e-5)


@pytest.mark.parametrize('new_len', [120, 36])
@pytest.mark.parametrize('mag_spec', [
    torch.randn(1, 257, 391),
    torch.randn(1, 2, 257, 391),
])
def test_ApplyFilterbank(mag_spec, new_len):
    """
    Test ApplyFilterbank to transpose input before applying filter.
    """
    filterbank = torch.randn(mag_spec.size(-2), new_len)
    apply_filterbank = ApplyFilterbank(filterbank)
    mag_spec_filterbanked = apply_filterbank(mag_spec)

    assert mag_spec.size(-1) == mag_spec_filterbanked.size(-1)
    assert mag_spec_filterbanked.size(-2) == new_len
    assert mag_spec.dim() == mag_spec_filterbanked.dim()


@pytest.mark.parametrize('amplitude,db', [
    (torch.Tensor([0.000001, 0.0001, 0.1, 1.0, 10.0, 1000000.0]),
     torch.Tensor([-60.0, -40.0, -10.0, 0.0, 10.0, 60.0]))
])
def test_amplitude_db(amplitude, db):
    """Test amplitude_to_db and db_to_amplitude."""
    amplitude_to_db = AmplitudeToDb(ref=1.0)
    db_to_amplitude = DbToAmplitude(ref=1.0)
    assert _approx_all_equal(db, amplitude_to_db(amplitude))
    assert _approx_all_equal(amplitude, db_to_amplitude(db))
    # both ways
    assert _approx_all_equal(db_to_amplitude(amplitude_to_db(amplitude)),
                             amplitude)
    assert _approx_all_equal(amplitude_to_db(db_to_amplitude(db)),
                             db,
                             atol=1e-5)


@pytest.mark.parametrize('rate', [0.5, 1.01, 1.3])
@pytest.mark.parametrize('complex_specgrams', [
    torch.randn(1, 2, 1025, 400, 2),
    torch.randn(1, 1025, 400, 2)
])
@pytest.mark.parametrize('hop_length', [256])
def test_TimeStretch(complex_specgrams, rate, hop_length):

    class use_double_precision:
        def __enter__(self):
            self.default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float64)

        def __exit__(self, type, value, traceback):
            torch.set_default_dtype(self.default_dtype)

    with use_double_precision():

        complex_specgrams = complex_specgrams.type(torch.get_default_dtype())
        # == Test shape
        layer = TimeStretch(hop_length=hop_length,
                            num_freqs=complex_specgrams.shape[-3],
                            fixed_rate=rate)
        complex_specgrams_stretch = layer(complex_specgrams)

        expected_size = list(complex_specgrams.size())
        expected_size[-2] = int(np.ceil(expected_size[-2] / rate))

        assert complex_specgrams.dim() == complex_specgrams_stretch.dim()
        assert complex_specgrams_stretch.size() == torch.Size(expected_size)

        # == Test values
        index = [0] * (complex_specgrams.dim() - 3) + [slice(None)] * 3
        mono_complex_specgram = complex_specgrams[index].numpy()
        mono_complex_specgram = mono_complex_specgram[..., 0] + \
            mono_complex_specgram[..., 1] * 1j
        expected_complex_stretch = librosa.core.phase_vocoder(
            mono_complex_specgram,
            rate=rate,
            hop_length=hop_length)

        complex_stretch = complex_specgrams_stretch[index].numpy()
        complex_stretch = complex_stretch[..., 0] + \
            1j * complex_stretch[..., 1]
        assert np.allclose(complex_stretch,
                           expected_complex_stretch, atol=1e-5)


class Tester(unittest.TestCase):

    def test_ComplexNorm(self):
        """
        ComplexNorm should normalize input correctly given a power of the magnitude.
        """

        def _test_powers(p):
            _seed()
            complex_spec = torch.randn(1, 257, 391, 2)
            complex_norm = ComplexNorm(power=p)
            mag_spec = complex_norm(complex_spec)
            mag_spec_manual = complex_spec[0][0][0].pow(2).sum(-1).pow(1 / 2).pow(p)
            assert complex_spec.shape[1:3] == mag_spec.shape[1:3]
            assert mag_spec_manual == mag_spec[0][0][0]
            assert complex_spec.dim() - mag_spec.dim() == 1

        for p in [1., 2., 0.7]:
            _test_powers(p)

    def test_Spectrogram(self):
        """
        Spectrogram in an nn.Module should not store buffers in the state dict
        """

        def _create_toy_model(channels):
            _seed()
            fft_len, hop_len = 512, 256
            spectrogram_layer = Spectrogram(
                fft_length=fft_len, hop_length=hop_len, power=1)
            conv_layer = nn.Conv2d(channels, 16, 3)
            return nn.Sequential(spectrogram_layer, conv_layer)

        waveforms = torch.randn(1, 2, 100000)
        toy_model = _create_toy_model(waveforms.size(1))

        toy_model[1].weight.data.fill_(0)
        sd = toy_model.state_dict()

        toy_model2 = _create_toy_model(waveforms.size(1))
        toy_model2.load_state_dict(sd)

        assert len(sd.keys()) == 2
        assert toy_model2[1].weight.data.sum() == 0

    def test_Melspectrogram(self):
        """
        Melspectrogram in an nn.Module should not store buffers in the state dict.
        Melspectrogram should work with a custom MelFilterbank
        """

        def _create_toy_model(channels):
            _seed()
            num_mels, sample_rate, fft_length, hop_length = 96, 22050, 512, 256
            mel_layer = Melspectrogram(
                num_mels=num_mels,
                sample_rate=sample_rate,
                fft_length=fft_length,
                hop_length=hop_length)
            conv_layer = nn.Conv2d(channels, 16, 3)
            return nn.Sequential(mel_layer, conv_layer)

        class TestFilterbank(MelFilterbank):
            """
            Base class for providing a filterbank matrix.
            """

            def __init__(self, *args):
                super(TestFilterbank, self).__init__(*args)

            def get_filterbank(self):
                return torch.randn(self.num_freqs, self.num_mels)

        def _test_mel_sd():
            _seed()
            waveforms = torch.randn(1, 1, 100000)
            toy_model = _create_toy_model(waveforms.size(1))

            toy_model[1].weight.data.fill_(0)
            sd = toy_model.state_dict()

            toy_model2 = _create_toy_model(waveforms.size(1))
            toy_model2.load_state_dict(sd)

            assert len(sd.keys()) == 2
            assert toy_model2[1].weight.data.sum() == 0

        # def _test_custom_fb():
        #     _seed()
        #     num_mels, sample_rate, fft_len, hop_len = 128, 22050, 512, 256
        #     waveforms = torch.randn(1, 1, 100000)
        #     mel_layer = Melspectrogram(
        #         num_mels=num_mels,
        #         sample_rate=sample_rate,
        #         fft_len=fft_len,
        #         hop_len=hop_len,
        #         mel_filterbank=TestFilterbank)
        #     mel_spect = mel_layer(waveforms)
        #     assert mel_spect.size(-2) == num_mels

        _test_mel_sd()
        # _test_custom_fb()

    def test_mu_law(self):
        """test mu-law encoding and decoding"""

        def _test_mu_encoding():
            _seed()
            n_quantize = 256
            encoding_layer = MuLawEncoding(n_quantize)

            waveform = 2 * (torch.rand(1, 1024) - 0.5)  # in [-1, 1)
            # manual computation
            mu = torch.tensor(n_quantize - 1, dtype=waveform.dtype, requires_grad=False)
            waveform_mu = waveform.sign() * torch.log1p(mu * waveform.abs()) / torch.log1p(mu)
            waveform_mu = ((waveform_mu + 1) / 2 * mu + 0.5).long()

            assert _all_equal(encoding_layer(waveform),
                              waveform_mu)

        def _test_mu_decoding():
            _seed()
            n_quantize = 256
            decoding_layer = MuLawDecoding(n_quantize)

            waveform_mu = torch.randint(low=0, high=n_quantize - 1,
                                        size=(1, 1024))

            # manual computation
            waveform_mu = waveform_mu.to(torch.float)
            mu = torch.tensor(n_quantize - 1, dtype=waveform_mu.dtype,
                              requires_grad=False)  # confused about dtype here..
            waveform = (waveform_mu / mu) * 2 - 1.
            waveform = waveform.sign() * (torch.exp(waveform.abs() * torch.log1p(mu)) - 1.) / mu

            assert _all_equal(decoding_layer(waveform_mu),
                              waveform)

        def _test_both_ways():
            _seed()
            n_quantize = 256
            encoding_layer = MuLawEncoding(n_quantize)
            decoding_layer = MuLawDecoding(n_quantize)

            waveform_mu = torch.randint(low=0, high=n_quantize - 1,
                                        size=(1, 1024))
            assert _all_equal(waveform_mu,
                              encoding_layer(decoding_layer(waveform_mu)))

        _test_mu_encoding()
        _test_mu_decoding()
        _test_both_ways()


if __name__ == '__main__':
    unittest.main()
