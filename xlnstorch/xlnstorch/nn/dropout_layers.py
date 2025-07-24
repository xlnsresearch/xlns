import torch
from . import LNSModule

class LNSDropout(LNSModule):
    """
    An LNS dropout layer that randomly zeroes some of the elements of the input tensor
    with a probability :math:`p` during training. This is not strictly necessary, since
    there are no trainable parameters, but it is included for completeness.

    See also: :py:class:`torch.nn.Dropout`

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

class LNSDropout1d(LNSModule):
    r"""
    An LNS dropout layer that randomly zeroes out entire channels of the input tensor
    with a probability :math:`p` during training. This is not strictly necessary, since
    there are no trainable parameters, but it is included for completeness.

    A channel is a 1D feature map, e.g., the :math:`j`-th channel of the :math:`i`-th
    sample in the batched input is a 1D tensor :math:`\text{input}[i, j]`.

    See also: :py:class:`torch.nn.Dropout1d`

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
        return torch.nn.functional.dropout1d(x, self.p, self.training, self.inplace)

class LNSDropout2d(LNSModule):
    r"""
    An LNS dropout layer that randomly zeroes out entire channels of the input tensor
    with a probability :math:`p` during training. This is not strictly necessary, since
    there are no trainable parameters, but it is included for completeness.

    A channel is a 2D feature map, e.g., the :math:`j`-th channel of the :math:`i`-th
    sample in the batched input is a 2D tensor :math:`\text{input}[i, j]`.

    Note: Following PyTorch's convention, this class will perform 1D channel-wise dropout
    for 3D inputs.

    See also: :py:class:`torch.nn.Dropout2d`

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
        return torch.nn.functional.dropout2d(x, self.p, self.training, self.inplace)

class LNSDropout3d(LNSModule):
    r"""
    An LNS dropout layer that randomly zeroes out entire channels of the input tensor
    with a probability :math:`p` during training. This is not strictly necessary, since
    there are no trainable parameters, but it is included for completeness.

    A channel is a 3D feature map, e.g., the :math:`j`-th channel of the :math:`i`-th
    sample in the batched input is a 3D tensor :math:`\text{input}[i, j]`.

    See also: :py:class:`torch.nn.Dropout3d`

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
        return torch.nn.functional.dropout3d(x, self.p, self.training, self.inplace)