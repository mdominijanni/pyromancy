from typing import Any
import torch
import torch.nn as nn

from ..params import mparameters, set_dynamic_estep_params
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

    def __init__(self, *shape: int | None, **kwargs: Any) -> None:
        Node.__init__(self, *shape, **kwargs)
        self.bias = nn.Parameter(torch.empty(self.bshape), True)
        self._initshape = self.bias.unsqueeze(0).shape

        with torch.no_grad():
            self.bias.fill_(0.0)

    def activity(self) -> torch.Tensor:
        r"""Activity of the node.

        Returns:
            ~torch.Tensor: activity of the node.
        """
        return self.bias.unsqueeze(0).expand(self._initshape)

    def initialize(self, pred: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Initializes the node's state.

        Args:
            pred (~torch.Tensor): prediction for the basis of initialization.

        Returns:
            ~torch.Tensor: expanded bias vector.

        Tip:
            Only the shape of ``pred`` is used, so the tensor can use ``device="meta"``.
        """
        if not self.shapeobj.compat(*pred.shape):
            raise ValueError(
                f"shape specified by `pred` {(*pred.shape,)} "
                f"is incompatible with node shape {(*self.shapeobj,)}"
            )

        bias = self.bias.unsqueeze(0).expand_as(pred)
        self._initshape = bias.shape

        return bias

    def prediction(self, pred: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Expands the bias vector as the prediction for this node.

        Args:
            pred (~torch.Tensor): raw prediction from the parent.

        Returns:
            ~torch.Tensor: expanded bias vector.

        Tip:
            Only the shape of ``pred`` is used, so the tensor can use ``device="meta"``.
        """
        return self.bias.unsqueeze(0).expand_as(pred)

    def reset(self, **kwargs: Any) -> None:
        r"""Resets the node state."""
        self._initshape = self.bias.unsqueeze(0).shape


class InputNode(Node):
    r"""Node for provided input for predictive coding.

    Args:
        *shape (int | None): base shape of the node's state.
        trainable (bool, optional): if the node's state is updated on E-steps.
            Defaults to False.

    Attributes:
        value (~torch.nn.parameter.Parameter): current value of the node.

    Tip:
        Since ``InputNode`` does not have a defined energy function, it should have
        no upstream nodes.
    """

    value: nn.Parameter

    def __init__(
        self, *shape: int | None, trainable: bool = False, **kwargs: Any
    ) -> None:
        Node.__init__(self, *shape, **kwargs)

        self.value = nn.Parameter(torch.empty(0), trainable)

        if trainable:
            set_dynamic_estep_params(self, "value")

    def activity(self) -> nn.Parameter:
        r"""Activity of the node.

        Args:
            value (~torch.Tensor): value to set the activity to.

        Returns:
            ~torch.Tensor: activity of the node.
        """
        return self.value

    @torch.no_grad()
    def initialize(self, pred: torch.Tensor, **kwargs: Any) -> nn.Parameter:
        r"""Initializes the node's state.

        Args:
            pred (~torch.Tensor): predictions for the basis of initialization.

        Returns:
            ~torch.Tensor: the node's initial activity.
        """
        z = self.prediction(pred, **kwargs)

        if not self.shapeobj.compat(*z.shape):
            raise ValueError(
                f"shape specified by `pred` {(*z.shape,)} "
                f"is incompatible with node shape {(*self.shapeobj,)}"
            )

        self.value.data = self.value.data.new_empty(*z.shape)
        self.value.copy_(z)

        return self.value

    def prediction(self, pred: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Transforms a raw prediction into one compatible with the node.

        Args:
            pred (~torch.Tensor): raw prediction from the parent.

        Returns:
            ~torch.Tensor: expanded bias vector.

        Note:
            This is the identity function.
        """
        return pred

    @torch.no_grad()
    def reset(self, **kwargs: Any) -> None:
        r"""Resets the node state."""
        self.zero_grad()
        self.value.data = self.value.new_empty(0)
