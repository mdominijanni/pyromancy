from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from ..infra import Shape
from ..params import eparameters, mparameters


@eparameters()
@mparameters()
class Node(nn.Module, ABC):
    r"""Base class for predictive coding nodes.

    Args:
        *shape (int | None): base shape of the node's state.

    Important:
        A placeholder :math:`\text{0}^\text{th}` dimension is automatically added
        to ``shape``, corresponding to the batch dimension.
    """

    _shape: Shape

    def __init__(self, *shape: int | None, **kwargs: Any) -> None:
        nn.Module.__init__(self, **kwargs)
        self._shape = Shape(None, *shape)

    @property
    def shapeobj(self) -> Shape:
        r"""Object storing the node shape.

        Returns:
            Shape: object storing the node shape.
        """
        return self._shape

    @property
    def rshape(self) -> tuple[int | None, ...]:
        r"""Shape of the node state, excluding the batch dimension.

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
        r"""Shape of the node state, excluding the batch dimension, safe for tensor construction.

        Returns:
            tuple[int, ...]: shape of the node state.

        Note:
            Placeholder dimensions represented with unit length dimensions.
            Use :py:meth:`~pyromancy.nodes.base.Node.rshape` for a version to use
            that preserves placeholders.
        """
        return self._shape.bshape[1:]

    @property
    def size(self) -> int:
        r"""Size of the node state.

        Returns:
            int: size of the node state.

        Note:
            This size only includes fixed dimensions, excluding filled placeholders.
        """
        return self._shape.size

    @abstractmethod
    def activity(self, **kwargs) -> torch.Tensor:
        r"""Activity of the node.

        Returns:
            ~torch.Tensor: activity of the node.
            **kwargs (~typing.Any): subclass-specific keyword arguments.

        Raises:
            NotImplementedError: must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize(self, pred: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Initializes the node's state.

        Args:
            pred (~torch.Tensor): prediction for the basis of initialization.
            **kwargs (~typing.Any): subclass-specific keyword arguments.

        Returns:
            ~torch.Tensor: the node's initial activity.

        Raises:
            NotImplementedError: must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def prediction(self, pred: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Transforms a raw prediction into one compatible with the node.

        Args:
            pred (~torch.Tensor): raw prediction from the parent.
            **kwargs (~typing.Any): subclass-specific keyword arguments.

        Raises:
            NotImplementedError: must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, **kwargs: Any) -> None:
        r"""Resets transient node state.

        Args:
            **kwargs (~typing.Any): subclass-specific keyword arguments.

        Raises:
            NotImplementedError: must be implemented by subclasses.

        Tip:
            This operation is typically executed after each new batch. With inference
            learning, this is done after each M-step. With incremental inference
            learning, this is done after the *final* M-step.
        """
        raise NotImplementedError

    def forward(self, pred: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Computes a forward pass on the node.

        When ``self.training`` is True, the node is initialized to the prediction and its
        activity is returned. When ``self.training`` is False, the prediction is directly
        returned.

        Args:
            pred (~torch.Tensor): prediction of the node's activity.
            **kwargs (~typing.Any): subclass-specific keyword arguments.

        Returns:
            ~torch.Tensor: activity of the node.

        Note:
            Unless overridden by the subclass, keyword arguments are passed to
            ``self.initialize()`` when ``self.training`` is True,
            and to ``self.prediction()`` when ``self.training`` is False.
        """
        if self.training:
            return self.initialize(pred, **kwargs)
        else:
            return self.prediction(pred, **kwargs)


class PredictiveNode(Node, ABC):
    r"""Base class for predictive coding nodes with a defined energy function.

    Args:
        *shape (int | None): base shape of the node's state.
    """

    def __init__(self, *shape: int | None, **kwargs: Any) -> None:
        Node.__init__(self, *shape, **kwargs)

    @abstractmethod
    def energy(self, pred: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Computes batchwise energy between a prediction and the node's activity.

        Args:
            pred (~torch.Tensor): prediction of the node's activity.
            **kwargs (~typing.Any): subclass-specific keyword arguments.

        Returns:
            ~torch.Tensor: batchwise energy between the activity and the prediction.

        Raises:
            NotImplementedError: must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def error(self, pred: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Computes elementwise error between a prediction and the node's activity.

        Args:
            pred (~torch.Tensor): prediction of the node's activity.
            **kwargs (~typing.Any): subclass-specific keyword arguments.

        Returns:
            ~torch.Tensor: elementwise error between the activity and the predictions.

        Raises:
            NotImplementedError: must be implemented by subclasses.
        """
        raise NotImplementedError


class VariationalNode(PredictiveNode, ABC):
    r"""Base class for predictive coding nodes modeling a well-formed distribution.

    Args:
        *shape (int | None): base shape of the node's state.
    """

    def __init__(self, *shape: int | None, **kwargs: Any) -> None:
        PredictiveNode.__init__(self, *shape, **kwargs)

    def initialize(
        self,
        pred: torch.Tensor,
        sample: bool = False,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Initializes the node's state.

        Args:
            pred (~torch.Tensor): prediction for the basis of initialization.
            sample (bool, optional): if the activity should be initialized based on a
                sampled probability rather than the MAP estimate. Defaults to False.
            generator (~torch.Generator | None, optional): pseudorandom number generator
                for sampling. Defaults to None.
            **kwargs (~typing.Any): subclass-specific keyword arguments.

        Returns:
            ~torch.Tensor: the node's initial activity.

        Raises:
            NotImplementedError: must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        pred: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Samples from the learned conditional distribution.

        Args:
            pred (~torch.Tensor | None, optional): prediction of the node's activity,
                used as parameters of the distribution. When None, the node's activity
                is used. Defaults to None.
            generator (~torch.Generator | None, optional): pseudorandom number generator
                for sampling. Defaults to None.
            **kwargs (~typing.Any): subclass-specific keyword arguments.

        Returns:
            ~torch.Tensor: samples from the conditional distribution.

        Raises:
            NotImplementedError: must be implemented by subclasses.
        """
        raise NotImplementedError

    def forward(
        self,
        pred: torch.Tensor,
        sample: bool = False,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Computes a forward pass on the node.

        When ``self.training`` is True, the node is initialized to the prediction and its
        activity is returned. When ``self.training`` is False, the prediction is directly
        returned.

        Args:
            pred (~torch.Tensor): prediction of the node's activity.
            sample (bool, optional): if the activity should be initialized based on a
                sampled probability rather than the MAP estimate. Defaults to False.
            generator (~torch.Generator | None, optional): pseudorandom number generator
                for sampling. Defaults to None.
            **kwargs (~typing.Any): subclass-specific keyword arguments.

        Returns:
            ~torch.Tensor: activity of the node.

        Note:
            Unless overridden by the subclass, keyword arguments are passed to
            ``self.initialize()`` when ``self.training`` is True, and to
            ``self.prediction()`` or ``self.sample()` when ``self.training`` is False.
        """
        if self.training:
            return self.initialize(pred, sample=sample, generator=generator, **kwargs)
        elif sample:
            return self.sample(pred, generator=generator, **kwargs)
        else:
            return self.prediction(pred, **kwargs)
