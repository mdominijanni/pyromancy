import torch
import torch.nn as nn

from ..utils import eparameters, mparameters
from .base import Node


@mparameters("bias")
class BiasNode(Node):
    r"""Trainable bias node for unsupervised predictive coding.

    Args:
        *shape (int | None): shape of the learned bias.

    Attributes:
        bias (~torch.nn.parameter.Parameter): learned bias :math:`\mathbf{b}`.
    """

    bias: nn.Parameter
    _initshape: torch.Size

    def __init__(self, *shape: int | None) -> None:
        Node.__init__(self, *shape)
        self.bias = nn.Parameter(torch.empty(self.bshape), True)
        self._initshape = self.bias.unsqueeze(0).shape

        with torch.no_grad():
            self.bias.fill_(0.0)

    @property
    def activity(self) -> torch.Tensor:
        r"""Activity of the node.

        Returns:
            ~torch.Tensor: activity (state) of the node.
        """
        return self.bias.unsqueeze(0).expand(self._initshape)

    def reset(self) -> None:
        r"""Resets transient node state."""
        self._initshape = self.bias.unsqueeze(0).shape

    def init(self, value: torch.Tensor) -> torch.Tensor:
        r"""Initializes the node's returned bias with a new shape.

        Args:
            value (~torch.Tensor): tensor to use as the shape for the returned bias.

        Returns:
            ~torch.Tensor: the expanded bias

        Raises:
            RuntimeError: shape of ``value`` is incompatible with the node.
        """
        if not self.shapeobj.compat(*value.shape):
            raise ValueError(
                f"shape of `value` {(*value.shape,)} is incompatible "
                f"with node shape {(*self.shapeobj,)}"
            )

        self._initshape = value.shape

        return self.bias.unsqueeze(0).expand(self._initshape)

    def error_from(self, value: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        r"""Computes elementwise error for a prediction of the node state and its presumed state.

        .. math::
            \boldsymbol{\varepsilon} = \mathbf{b} - \boldsymbol{\mu}

        Args:
            value (~torch.Tensor): presumed value of the node state :math:`\mathbf{b}`.
            pred (~torch.Tensor): predicted bias :math:`\boldsymbol{\mu}`.

        Returns:
            ~torch.Tensor: elementwise error :math:`\boldsymbol{\varepsilon}`.
        """
        return value - pred

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Expands bias tensor for the network.

        Args:
            inputs (~torch.Tensor): tensor to use as the shape for the returned bias.

        Returns:
            ~torch.Tensor: expanded bias tensor.

        Tip:
            ``inputs`` should have the desired shape (including the batch dimension)
            to use the returned bias as a prediction for initialization/inference. The
            contents, device, and data type of ``inputs`` are unused.
        """
        if self.training:
            return self.init(inputs)
        else:
            return self.bias.unsqueeze(0).expand_as(inputs)


class FixedNode(Node):
    r"""Input node with a fixed value.

    Args:
        *shape (int | None): base shape of the node's state.

    Attributes:
        value (~torch.nn.parameter.Buffer): current value of the node.

    Hint:
        This is primarily useful when performing *query by conditioning* from an input,
        where the value is not updated on E-steps.
    """

    value: nn.Buffer

    def __init__(self, *shape: int | None) -> None:
        Node.__init__(self, *shape)
        self.value = nn.Buffer(torch.empty(0))

    @property
    def activity(self) -> nn.Buffer:
        r"""Activity of the node.

        Returns:
            ~torch.nn.Buffer: activity (state) of the node.
        """
        return self.value

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
            value (~torch.Tensor): value to initialize to.

        Returns:
            ~torch.nn.parameter.Buffer: the reinitialized value.

        Raises:
            RuntimeError: shape of ``value`` is incompatible with the node.
        """
        if not self.shapeobj.compat(*value.shape):
            raise ValueError(
                f"shape of `value` {(*value.shape,)} is incompatible "
                f"with node shape {(*self.shapeobj,)}"
            )

        self.value.data = self.value.data.new_empty(*value.shape)
        self.value.copy_(value)

        return self.value

    def error_from(self, value: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        r"""Computes elementwise error for a prediction of the node state and its presumed state.

        .. math::
            \boldsymbol{\varepsilon} = \mathbf{z} - \boldsymbol{\mu}

        Args:
            value (~torch.Tensor): presumed value of the node state :math:`\mathbf{z}`.
            pred (~torch.Tensor): predicted value :math:`\boldsymbol{\mu}`.

        Returns:
            ~torch.Tensor: elementwise error :math:`\boldsymbol{\varepsilon}`.
        """
        return value - pred

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Computes a forward pass on the node.

        When ``self.training`` is True, the prediction is assigned to the value and then
        value is returned. When ``self.training`` is False, the prediction is directly
        returned (i.e. this acts as the identity operation).

        Args:
            inputs (~torch.Tensor): prediction of the value.

        Returns:
            ~torch.Tensor: value of the node.
        """
        if self.training:
            return self.init(inputs)
        else:
            return inputs


@eparameters("value")
class FloatNode(Node):
    r"""Input node with an trainable value.

    Args:
        *shape (int | None): base shape of the node's state.

    Attributes:
        value (~torch.nn.parameter.Parameter): current value of the node.

    Hint:
        This is primarily useful when performing *query by initialization* from an input,
        where the value is updated on E-steps.
    """

    value: nn.Parameter

    def __init__(self, *shape: int | None) -> None:
        Node.__init__(self, *shape)
        self.value = nn.Parameter(torch.empty(0), True)

    @property
    def activity(self) -> nn.Parameter:
        r"""Activity of the node.

        Returns:
            ~torch.nn.Parameter: activity (state) of the node.
        """
        return self.value

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
    def init(self, value: torch.Tensor) -> nn.Parameter:
        r"""Initializes the node's state to a new value.

        Args:
            value (~torch.Tensor): value to initialize to.

        Returns:
            ~torch.nn.parameter.Parameter: the reinitialized value.

        Raises:
            RuntimeError: shape of ``value`` is incompatible with the node.
        """
        if not self.shapeobj.compat(*value.shape):
            raise ValueError(
                f"shape of `value` {(*value.shape,)} is incompatible "
                f"with node shape {(*self.shapeobj,)}"
            )

        self.value.data = self.value.data.new_empty(*value.shape)
        self.value.copy_(value)

        return self.value

    def error_from(self, value: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        r"""Computes elementwise error for a prediction of the node state and its presumed state.

        .. math::
            \boldsymbol{\varepsilon} = \mathbf{z} - \boldsymbol{\mu}

        Args:
            value (~torch.Tensor): presumed value of the node state :math:`\mathbf{z}`.
            pred (~torch.Tensor): predicted value :math:`\boldsymbol{\mu}`.

        Returns:
            ~torch.Tensor: elementwise error :math:`\boldsymbol{\varepsilon}`.
        """
        return value - pred

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Computes a forward pass on the node.

        When ``self.training`` is True, the prediction is assigned to the value and then
        value is returned. When ``self.training`` is False, the prediction is directly
        returned (i.e. this acts as the identity operation).

        Args:
            inputs (~torch.Tensor): prediction of the value.

        Returns:
            ~torch.Tensor: value of the node.
        """
        if self.training:
            return self.init(inputs)
        else:
            return inputs
