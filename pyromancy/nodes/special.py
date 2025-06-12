import torch
import torch.nn as nn
from ..utils import mparameters
from .base import Node


@mparameters("bias")
class Bias(Node):
    r"""Trainable bias node for unsupervised predictive coding.

    Args:
        *shape (int | None): shape of the learned bias.

    Attributes:
        bias (nn.Parameter): learned bias :math:`\mu`.
    """

    bias: nn.Parameter

    def __init__(self, *shape: int | None) -> None:
        nn.Module.__init__(self, *shape)
        self.bias = nn.Parameter(torch.empty(self.bshape), True)

        with torch.no_grad():
            self.bias.fill_(0.0)

    def reset(self) -> None:
        r"""Resets transient node state."""
        pass

    def forward(self, batch_size: int, *fill: int) -> torch.Tensor:
        r"""Expands bias tensor for network.

        Args:
            batch_size (int): target batch size.
            *fill (int): additional sizes to fill placeholder dimensions.

        Returns:
            torch.Tensor: expanded bias tensor.
        """
        return self.bias.unsqueeze(0).expand(self.shapeobj.filled(batch_size, *fill))
