import torch
import torch.nn as nn
import torch.nn.functional as F


class AmplitudeToDb(nn.Module):
    """A layer that applies decibel mapping (10-base logarithmic scaling)"""

    def __init__(self, ref=1.0, amin=1e-7):
        """

        :param ref: Reference value of the input that would result in `0.0` after this conversion
        :param amin:
        """
        super(AmplitudeToDb, self).__init__()
        self.ref = nn.Parameter(ref, requires_grad=False)
        self.amin = nn.Parameter(amin, requires_grad=False)
        assert ref > amin, "Reference value is expected to be bigger than amin, but I have" \
                           "ref:{} and amin:{}".format(ref, amin)

    def forward(self, x):
        """
        :param x: torch.tensor, expectedly non-negative.
        :return: decibel-scaled x in the same shape
        """
        return amplitude_to_db(x, ref=self.ref, amin=self.amin)


def amplitude_to_db(x, ref=1.0, amin=1e-7):
    """
    Note: Given that FP32 is used and its corresponding `amin`,
    we do not implement further numerical stabilization for very small inputs.

    :param x: torch.tensor, the input value
    :param ref:
    :param amin: float
    :return: torch.tensor, same size of x, but decibel-scaled
    """
    x = torch.clamp(x, min=amin)
    return 10.0 * (torch.log10(x) - torch.log10(torch.tensor(ref, device=x.device, requires_grad=False)))


class DbToAmplitude():
    """A layer that applies *inverse* decibel mapping (10-base logarithmic scaling)"""

    def __init__(self, ref=1.0):
        """

        :param ref:
        """
        super(DbToAmplitude, self).__init__()
        self.ref = nn.Parameter(ref, requires_grad=False)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        return db_to_amplitude(x, ref=self.ref)


def db_to_amplitude(x, ref=1.0):
    """

    :param x:
    :param ref:
    :return:
    """
    return torch.pow(10.0, x / 10.0 + torch.log10(torch.tensor(ref, device=x.device, requires_grad=False)))


class MuLawEncoding(nn.Module):
    """Apply mu-law encoding to the input tensor"""

    def __init__(self, n_quantize=256):
        super(MuLawEncoding, self).__init__()
        self.nq = nn.Parameter(n_quantize)

    def forward(self, x):
        """

        :param x: torch.tensor
        :return:
        """
        return mu_law_encoding(x, self.nq)


def mu_law_encoding(x, n_quantize=256):
    if not x.dtype.is_floating_point:
        x = x.to(torch.float)
    mu = torch.tensor(n_quantize - 1, dtype=x.dtype, requires_grad=False)  # confused about dtype here..

    x_mu = x.sign() * torch.log1p(mu * x.abs()) / torch.log1p(mu)
    x_mu = ((x_mu + 1) / 2 * mu + 0.5).long()
    return x_mu


class MuLawDecoding(nn.Module):
    """Apply mu-law decoding (expanding) to the input tensor"""

    def __init__(self, n_quantize=256):
        super(MuLawDecoding, self).__init__()
        self.nq = nn.Parameter(n_quantize)

    def forward(self, x_mu):
        return mu_law_decoding(x_mu, self.nq)


def mu_law_decoding(x_mu, n_quantize=256):
    if not x_mu.dtype.is_floating_point:
        x_mu = x_mu.to(torch.float)
    mu = torch.tensor(n_quantize - 1, dtype=x_mu.dtype, requires_grad=False)  # confused about dtype here..
    x = ((x_mu) / mu) * 2 - 1.
    x = x.sign() * (torch.exp(x.abs() * torch.log1p(mu)) - 1.) / mu
    return x


if __name__ == "__main__":
    # a super quick test for amplitude <-> db
    x = torch.tensor([0.1, 1., 10., 100., 1000.])
    print("Input x: ", x)
    x_db = amplitude_to_db(x)
    print("  * decibel: ", x_db)
    x_recon = db_to_amplitude(x_db)
    print("  * recon  : ", x_recon)

    # test for mu-law encoding <-> decoding
    x = torch.tensor([-1, -0.1, -0.001, 0, 0.1, 1])
    print("Input x: ", x)
    x_mu = mu_law_encoding(x)
    print("  * x_mu   : ", x_mu)
    x_recon = mu_law_decoding(x_mu)
    print("  * x recon:", x_recon)
