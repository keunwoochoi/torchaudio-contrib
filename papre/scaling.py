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
        assert ref > amin, "Reference value is expected to be bigger than"

    def forward(self, x):
        """
        :param x: torch.tensor, expectedly non-negative.
        :return: decibel-scaled x in the same shape
        """
        return amplitude_to_db(x, self.ref, self.amin)


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
    return 10.0 * (torch.log10(x) - torch.log10(torch.tensor(ref, device=x.device)))
