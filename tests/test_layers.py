"""
Test the layers. Currently only on cpu since travis doesn't have GPU.
"""
import unittest
import torch
import torch.nn as nn
from torchaudio_contrib.layers import STFT, ComplexNorm, \
    ApplyFilterbank, Spectrogram, Melspectrogram, MelFilterbank, \
    AmplitudeToDb, DbToAmplitude, MuLawEncoding, MuLawDecoding


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

        def _test_mono_cpu():
            _seed()
            signal = torch.randn(1, 100000)
            fft_len, hop_len = 512, 256
            layer = STFT(fft_len=fft_len, hop_len=hop_len)
            spect = layer(signal)
            assert spect.size(0) == 1
            assert spect.size(1) == fft_len // 2 + 1
            assert spect.size(2) == _num_stft_bins(
                signal.size(-1), fft_len, hop_len, fft_len // 2)
            assert spect.dim() == signal.dim() + 2

        def _test_batch_multichannel_cpu():
            _seed()
            signal = torch.randn(1, 2, 100000)
            fft_len, hop_len = 512, 256
            layer = STFT(fft_len=fft_len, hop_len=hop_len)
            spect = layer(signal)

            assert spect.size(1) == signal.size(1)
            assert spect.dim() == signal.dim() + 2

        _test_mono_cpu()
        _test_batch_multichannel_cpu()

    def test_ComplexNorm(self):
        """
        ComplexNorm should normalize input correctly given a power of the magnitude.
        """

        def _test_powers(p):
            _seed()
            stft = torch.randn(1, 257, 391, 2)
            layer = ComplexNorm(power=p)
            spect = layer(stft)
            manual_spect = stft[0][0][0].pow(2).sum(-1).pow(1 / 2).pow(p)
            assert stft.shape[1:3] == spect.shape[1:3]
            assert manual_spect == spect[0][0][0]
            assert stft.dim() - spect.dim() == 1

        for p in [1., 2., 0.7]:
            _test_powers(p)

    def test_ApplyFilterbank(self):
        """
        ApplyFilterbank should apply correct transpose to input before multiplying it by the filter
        """

        def _test_mono_cpu():
            _seed()
            new_len = 120
            spect = torch.randn(1, 257, 391)
            filterbank = torch.randn(spect.size(-2), new_len)

            layer = ApplyFilterbank(filterbank)
            spect2 = layer(spect)

            assert spect.size(-1) == spect2.size(-1)
            assert spect2.size(-2) == new_len
            assert spect.dim() == spect2.dim()

        def _test_multichannel_cpu():
            _seed()
            new_len = 120
            spect = torch.randn(1, 2, 257, 391)
            filterbank = torch.randn(spect.size(-2), new_len)

            layer = ApplyFilterbank(filterbank)
            spect2 = layer(spect)

            assert spect2.size(-2) == new_len
            assert spect.dim() == spect2.dim()

        _test_mono_cpu()
        _test_multichannel_cpu()

    def test_Spectrogram(self):
        """
        Spectrogram in an nn.Module should not store buffers in the state dict
        """

        def _create_toy_model(channels):
            _seed()
            fft_len, hop_len = 512, 256
            spect_layer = Spectrogram(
                fft_len=fft_len, hop_len=hop_len, power=1)
            conv_layer = nn.Conv2d(channels, 16, 3)
            return nn.Sequential(spect_layer, conv_layer)

        signal = torch.randn(1, 2, 100000)
        toy_model = _create_toy_model(signal.size(1))

        toy_model[1].weight.data.fill_(0)
        sd = toy_model.state_dict()

        toy_model2 = _create_toy_model(signal.size(1))
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
            signal = torch.randn(1, 1, 100000)
            toy_model = _create_toy_model(signal.size(1))

            toy_model[1].weight.data.fill_(0)
            sd = toy_model.state_dict()

            toy_model2 = _create_toy_model(signal.size(1))
            toy_model2.load_state_dict(sd)

            assert len(sd.keys()) == 2
            assert toy_model2[1].weight.data.sum() == 0

        def _test_custom_fb():
            _seed()
            num_bands, sample_rate, fft_len, hop_len = 128, 22050, 512, 256
            signal = torch.randn(1, 1, 100000)
            mel_layer = Melspectrogram(
                num_bands=num_bands,
                sample_rate=sample_rate,
                fft_len=fft_len,
                hop_len=hop_len,
                mel_filterbank=TestFilterbank)
            mel_spect = mel_layer(signal)
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

            x = 2 * (torch.rand(1, 1024) - 0.5)  # in [-1, 1)
            # manual computation
            mu = torch.tensor(n_quantize - 1, dtype=x.dtype, requires_grad=False)
            x_mu = x.sign() * torch.log1p(mu * x.abs()) / torch.log1p(mu)
            x_mu = ((x_mu + 1) / 2 * mu + 0.5).long()

            assert _all_equal(encoding_layer(x),
                              x_mu)

        def _test_mu_decoding():
            _seed()
            n_quantize = 256
            decoding_layer = MuLawDecoding(n_quantize)

            x_mu = torch.randint(low=0, high=n_quantize - 1,
                                 size=(1, 1024))

            # manual computation
            x_mu = x_mu.to(torch.float)
            mu = torch.tensor(n_quantize - 1, dtype=x_mu.dtype, requires_grad=False)  # confused about dtype here..
            x = (x_mu / mu) * 2 - 1.
            x = x.sign() * (torch.exp(x.abs() * torch.log1p(mu)) - 1.) / mu

            assert _all_equal(decoding_layer(x_mu),
                              x)

        def _test_both_ways():
            _seed()
            n_quantize = 256
            encoding_layer = MuLawEncoding(n_quantize)
            decoding_layer = MuLawDecoding(n_quantize)

            x_mu = torch.randint(low=0, high=n_quantize - 1,
                                 size=(1, 1024))
            assert _all_equal(x_mu,
                              encoding_layer(decoding_layer(x_mu)))

        _test_mu_encoding()
        _test_mu_decoding()
        _test_both_ways()


if __name__ == '__main__':
    unittest.main()
