"""
Test the layers. Currently only on cpu since travis doesn't have GPU.
"""
import torch
import torch.nn as nn
from torchaudio_contrib.layers import STFT, ComplexNorm, \
    ApplyFilterbank, Spectrogram, Melspectrogram, MelFilterbank, \
    AmplitudeToDb, DbToAmplitude, MuLawEncoding, MuLawDecoding
import unittest


def _num_stft_bins(signal_len, fft_len, hop_len, pad):
    return (signal_len + 2 * pad - fft_len + hop_len) // hop_len


def _seed(seed=1234):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def _approx_all_equal(x, y, atol=1e-7):
    return torch.all(torch.lt(torch.abs(torch.add(x, -y)), atol))


def _all_equal(x, y):
    return torch.all(torch.eq(x, y))


class Tester(unittest.TestCase):

    def test_STFT(self):
        """
        STFT should handle mutlichannel signal correctly

        Padding: Value in having padding outside of torch.stft?
        """

        def _test_mono_sizes_cpu():
            _seed()
            waveform = torch.randn(1, 100000)
            fft_len, hop_len = 512, 256
            layer = STFT(fft_len=fft_len, hop_len=hop_len)
            complex_spec = layer(waveform)
            assert complex_spec.size(0) == 1
            assert complex_spec.size(1) == fft_len // 2 + 1
            assert complex_spec.size(2) == _num_stft_bins(
                waveform.size(-1), fft_len, hop_len, fft_len // 2)
            assert complex_spec.dim() == waveform.dim() + 2

        def _test_batch_multichannel_sizes_cpu():
            _seed()
            waveform = torch.randn(1, 2, 100000)
            fft_len, hop_len = 512, 256
            layer = STFT(fft_len=fft_len, hop_len=hop_len)
            complex_spec = layer(waveform)

            assert complex_spec.size(1) == waveform.size(1)
            assert complex_spec.dim() == waveform.dim() + 2

        def _test_values():
            from librosa import stft as librosa_stft
            from librosa import magphase as librosa_magphase
            from numpy import allclose as np_allclose
            from torchaudio_contrib.functional import magphase

            _seed()
            waveform = torch.randn(1, 10000)
            fft_len, hop_len = 512, 256
            layer = STFT(fft_len=fft_len, hop_len=hop_len)
            complex_spec_torch = layer(waveform)  # (1, 257, 40, 2)
            complex_spec_librosa = librosa_stft(waveform.numpy()[0],
                                                n_fft=fft_len,
                                                hop_length=hop_len)

            mag_spec_torch, phase_spec_torch = magphase(complex_spec_torch)
            mag_spec_librosa, phase_spec_librosa = librosa_magphase(complex_spec_librosa)

            mag_spec_torch = mag_spec_torch.numpy()

            complex_spec_torch = complex_spec_torch.numpy()
            complex_spec_torch = complex_spec_torch[:, :, :, 0] + 1j * complex_spec_torch[:, :, :, 1]  # np.complex

            assert np_allclose(mag_spec_torch, mag_spec_librosa, atol=1e-6)
            assert np_allclose(complex_spec_torch, complex_spec_librosa, atol=1e-5)

        _test_mono_sizes_cpu()
        _test_batch_multichannel_sizes_cpu()
        # _test_values()

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

    def test_ApplyFilterbank(self):
        """
        ApplyFilterbank should apply correct transpose to input before multiplying it by the filter
        """

        def _test_mono_cpu():
            _seed()
            new_len = 120
            mag_spec = torch.randn(1, 257, 391)
            filterbank = torch.randn(mag_spec.size(-2), new_len)

            apply_filterbank = ApplyFilterbank(filterbank)
            mag_spec_filterbanked = apply_filterbank(mag_spec)

            assert mag_spec.size(-1) == mag_spec_filterbanked.size(-1)
            assert mag_spec_filterbanked.size(-2) == new_len
            assert mag_spec.dim() == mag_spec_filterbanked.dim()

        def _test_multichannel_cpu():
            _seed()
            new_len = 120
            mag_specs = torch.randn(1, 2, 257, 391)
            filterbank = torch.randn(mag_specs.size(-2), new_len)

            apply_filterbank = ApplyFilterbank(filterbank)
            mag_specs_filterbanked = apply_filterbank(mag_specs)

            assert mag_specs_filterbanked.size(-2) == new_len
            assert mag_specs.dim() == mag_specs_filterbanked.dim()

        _test_mono_cpu()
        _test_multichannel_cpu()

    def test_Spectrogram(self):
        """
        Spectrogram in an nn.Module should not store buffers in the state dict
        """

        def _create_toy_model(channels):
            _seed()
            fft_len, hop_len = 512, 256
            spectrogram_layer = Spectrogram(
                fft_len=fft_len, hop_len=hop_len, power=1)
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
            num_bands, sample_rate, fft_len, hop_len = 96, 22050, 512, 256
            mel_layer = Melspectrogram(
                num_bands=num_bands,
                sample_rate=sample_rate,
                fft_len=fft_len,
                hop_len=hop_len)
            conv_layer = nn.Conv2d(channels, 16, 3)
            return nn.Sequential(mel_layer, conv_layer)

        class TestFilterbank(MelFilterbank):
            """
            Base class for providing a filterbank matrix.
            """

            def __init__(self, *args):
                super(TestFilterbank, self).__init__(*args)

            def get_filterbank(self):
                return torch.randn(self.num_bins, self.num_bands)

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

        def _test_custom_fb():
            _seed()
            num_bands, sample_rate, fft_len, hop_len = 128, 22050, 512, 256
            waveforms = torch.randn(1, 1, 100000)
            mel_layer = Melspectrogram(
                num_bands=num_bands,
                sample_rate=sample_rate,
                fft_len=fft_len,
                hop_len=hop_len,
                mel_filterbank=TestFilterbank)
            mel_spect = mel_layer(waveforms)
            assert mel_spect.size(-2) == num_bands

        _test_mel_sd()
        _test_custom_fb()

    def test_amplitude_db(self):
        """test amplitude_to_db and db_to_amplitude"""

        def _test_amplitude_to_db():
            conversion_layer = AmplitudeToDb(ref=1.0)
            amplitude = [0.01, 0.1, 1.0, 10.0]
            db = [-20.0, -10.0, 0.0, 10.0]
            assert _approx_all_equal(torch.Tensor(db),
                                     conversion_layer(torch.Tensor(amplitude)))

        def _test_db_to_amplitude():
            conversion_layer = DbToAmplitude(ref=1.0)
            amplitude = [0.000001, 0.0001, 0.1, 1.0, 10.0, 1000000.0]
            db = [-60, -40.0, -10.0, 0.0, 10.0, 60.]
            assert _approx_all_equal(torch.Tensor(amplitude),
                                     conversion_layer(torch.Tensor(db)))

        def _test_both_ways():
            _seed()
            amplitude = torch.rand(1, 1024) + 1e-7
            db = 120 * torch.rand(1, 1024) - 60  # in [-60, 60]
            amplitude_to_db = AmplitudeToDb()
            db_to_amplitude = DbToAmplitude()

            assert _approx_all_equal(db_to_amplitude(amplitude_to_db(amplitude)),
                                     amplitude)
            assert _approx_all_equal(amplitude_to_db(db_to_amplitude(db)),
                                     db,
                                     atol=1e-5)

        _test_amplitude_to_db()
        _test_db_to_amplitude()
        _test_both_ways()

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
