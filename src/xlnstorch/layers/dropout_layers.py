import torch
from . import LNSModule

class LNSDropout(LNSModule):
    """
    An LNS dropout layer that randomly zeroes some of the elements of the input tensor
    with a probability `p` during training. This is not strictly necessary, since there
    are no trainable parameters, but it is included for completeness.

    See also: `torch.nn.Dropout`

    Parameters
    ----------
    p : float, optional
        The probability of an element to be zeroed. Default is 0.5.
    inplace : bool, optional
        If True, performs the operation in-place. Default is False.
    """

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.dropout(x, self.p, self.training, self.inplace)