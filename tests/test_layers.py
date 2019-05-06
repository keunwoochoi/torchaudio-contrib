"""
Test the layers. Currently only on cpu since travis doesn't have GPU.
"""
import unittest
import torch
import torch.nn as nn
from torchaudio_contrib.layers import STFT, ComplexNorm, \
    ApplyFilterbank, Spectrogram, Melspectrogram, MelFilterbank


def _num_stft_bins(signal_len, fft_len, hop_len, pad):
    return (signal_len + 2 * pad - fft_len + hop_len) // hop_len


def _seed():
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
        torch.backends.cudnn.deterministic = True


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


if __name__ == '__main__':
    unittest.main()
