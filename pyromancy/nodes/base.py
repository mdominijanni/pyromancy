from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..infra import Shape
from ..utils import eparameters, mparameters


@eparameters()
@mparameters()
class Node(nn.Module, ABC):
    r"""Base class for predictive coding nodes.

    Args:
        *shape (int | None): base shape of the node's state.

    Important:
        A placeholder :math:`\text{0}^\text{th}` dimension is automatically added
        to ``shape``.
    """

    _shape: Shape

    def __init__(self, *shape: int | None) -> None:
        nn.Module.__init__(self)
        self._shape = Shape(None, *shape)

    @property
    def shapeobj(self) -> Shape:
        r"""Object storing the node shape.

        Returns:
            Shape: object storing the node shape.
        """
        return self._shape

    @property
    def shape(self) -> tuple[int | None, ...]:
        r"""Shape of the node state.

        Returns:
            tuple[int | None, ...]: shape of the node state.

        Note:
            Placeholder dimensions represented with ``None`` values.
            Use :py:meth:`~pyromancy.nodes.base.Node.bshape` for a version to use
            when constructing broadcastable tensors.
        """
        return self._shape.rshape[1:]

    @property
    def bshape(self) -> tuple[int, ...]:
        r"""Shape of the node state, safe for tensor construction.

        Returns:
            tuple[int, ...]: shape of the node state.

        Note:
            Placeholder dimensions represented with unit length dimensions.
            Use :py:meth:`~pyromancy.nodes.base.Node.shape` for a version to use
            that preserves placeholders.
        """
        return self._shape.bshape[1:]

    @property
    def size(self) -> int:
        r"""Size of the node state.

        Returns:
            int: size of the node state.
        """
        return self._shape.size

    @abstractmethod
    def reset(self) -> None:
        r"""Resets transient node state.

        Raises:
            NotImplementedError: must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Computes a forward pass on the node.

        Args:
            inputs (~torch.Tensor): input to the node.

        Returns:
            ~torch.Tensor: value of the node.

        Raises:
            NotImplementedError: must be implemented by subclasses.

        Important:
            Subclasses implementing this method should perform the following operations:
            - Initialize the value of the node based on ``inputs`` if ``self.training`` is ``True``.
            - Return the value of the node.

            Most subclasses should inherit from :py:class:`~pyromancy.nodes.PredictiveNode`
            instead, although special cases may inherit from this class instead (see
            :py:class:`~pyromancy.nodes.BiasNode` for an example of this).
        """
        raise NotImplementedError


@eparameters("value")
class PredictiveNode(Node, ABC):
    r"""Base class for predictive coding nodes that generate predictions.

    Args:
        *shape (int | None): base shape of the node's state.

    Attributes:
        value (~torch.nn.parameter.Parameter): current value of the node.
    """

    value: nn.Parameter

    def __init__(self, *shape: int | None) -> None:
        Node.__init__(self, *shape)
        self.value = nn.Parameter(torch.empty(0), True)

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
            ValueError: shape of ``value`` is incompatible with the node.
        """
        if not self.shapeobj.compat(*value.shape):
            raise ValueError(
                f"shape of `value` {(*value.shape,)} is incompatible "
                f"with node shape {(*self.shapeobj,)}"
            )

        self.value.data = self.value.data.new_empty(*value.shape)
        self.value.copy_(value)

        return self.value

    @abstractmethod
    def error(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Computes elementwise error for a prediction of the node state.

        Args:
            pred (~torch.Tensor): prediction of the node state.

        Raises:
            NotImplementedError: must be implemented by subclasses.

        Returns:
            ~torch.Tensor: elementwise error between the state and a prediction.
        """
        raise NotImplementedError

    @abstractmethod
    def energy(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Computes variational free energy for a prediction of the node state.

        Args:
            pred (~torch.Tensor): prediction of the node state.

        Raises:
            NotImplementedError: must be implemented by subclasses.

        Returns:
            ~torch.Tensor: variational free energy between the state and a prediction.
        """
        raise NotImplementedError

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


class VariationalNode(PredictiveNode, ABC):
    r"""Base class for predictive coding nodes modelling a variational distribution.

    Args:
        *shape (int | None): base shape of the node's state.
    """

    def __init__(self, *shape: int | None) -> None:
        PredictiveNode.__init__(self, *shape)

    @abstractmethod
    def sample(
        self, value: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        r"""Samples from the learned variational distribution.

        Args:
            value (~torch.Tensor): location parameter of the variational distribution
                for sampling.
            generator (~torch.Generator | None, optional): pseudorandom number generator
                for sampling. Defaults to None.

        Raises:
            NotImplementedError: must be implemented by subclasses.

        Returns:
            ~torch.Tensor: samples from the variational distribution.
        """
        raise NotImplementedError
