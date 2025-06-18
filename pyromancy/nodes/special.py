import torch
import torch.nn as nn
from ..utils import mparameters
from .base import Node


@mparameters("bias")
class BiasNode(Node):
    r"""Trainable bias node for unsupervised predictive coding.

    Args:
        *shape (int | None): shape of the learned bias.

    Attributes:
        bias (nn.Parameter): learned bias :math:`\mathbf{b}`.
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

    def error(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Error between the prediction and node state.

        .. math::
            \boldsymbol{\varepsilon} = \mathbf{b} - \boldsymbol{\mu}

        Args:
            pred (torch.Tensor): predicted bias :math:`\boldsymbol{\mu}`.

        Returns:
            torch.Tensor: elementwise error :math:`\boldsymbol{\varepsilon}`.
        """
        return self.bias - pred

    def forward(self, batch_size: int, *fill: int) -> torch.Tensor:
        r"""Expands bias tensor for network.

        Args:
            batch_size (int): target batch size.
            *fill (int): additional sizes to fill placeholder dimensions.

        Returns:
            torch.Tensor: expanded bias tensor.
        """
        return self.bias.unsqueeze(0).expand(self.shapeobj.filled(batch_size, *fill))


class FixedNode(Node):
    r"""Fixed value input node.

    This is primarily useful for when it is helpful to have a dummy input node, but when
    its value should not be updated during E-steps (e.g. when using query by conditioning).

    Args:
        *shape (int | None): base shape of the node's state.

    Attributes:
        value (nn.Buffer): current value of the node.
    """

    value: nn.Buffer

    def __init__(self, *shape: int | None) -> None:
        Node.__init__(self, *shape)
        self.value = nn.Buffer(torch.empty(0), False)

    @torch.no_grad()
    def reset(self) -> None:
        r"""Resets the node state.

        This operation is typically executed after each new batch. With inference learning,
        this is done after M-step. With incremental inference learning, this is done after
        the *final* M-step.
        """
        self.zero_grad()
        self.value.data = self.value.new_empty(0)

    @torch.no_grad()
    def init(self, value: torch.Tensor) -> nn.Buffer:
        r"""Initializes the node's state to a new value.

        Args:
            value (torch.Tensor): value to initialize to.

        Returns:
            nn.Buffer: the reinitialized value.

        Raises:
            RuntimeError: shape of ``value`` is incompatible with the node.
        """

        assert self.shapeobj.compat(*value.shape)

        self.value.data = self.value.data.new_empty(*value.shape)
        self.value.copy_(value)

        return self.value

    def error(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Error between the prediction and node state.

        .. math::
            \boldsymbol{\varepsilon} = \mathbf{z} - \boldsymbol{\mu}

        Args:
            pred (torch.Tensor): predicted value :math:`\boldsymbol{\mu}`.

        Returns:
            torch.Tensor: elementwise error :math:`\boldsymbol{\varepsilon}`.
        """
        return self.value - pred

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Computes a forward pass on the node.

        When ``self.training`` is True, the prediction is assigned to the value and then
        value is returned. When ``self.training`` is False, the prediction is directly
        returned (i.e. this acts as the identity operation).

        Args:
            inputs (torch.Tensor): prediction of the value.

        Returns:
            torch.Tensor: value of the node.
        """
        if self.training:
            return self.init(inputs)
        else:
            return inputs
