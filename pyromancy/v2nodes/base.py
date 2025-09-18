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
    def shape(self) -> Shape:
        r"""Object storing the node shape.

        Returns:
            Shape: object storing the node shape.
        """
        return self._shape

    @property
    def size(self) -> int:
        r"""Size of the node state.

        Returns:
            int: size of the node state.

        Note:
            This size only includes fixed dimensions, excluding filled placeholders.
        """
        return self._shape.size

    @property
    @abstractmethod
    def activity(self) -> torch.Tensor:
        r"""Activity of the node.

        Args:
            value (~torch.Tensor): value to set the activity to.

        Returns:
            ~torch.Tensor: activity of the node.

        Raises:
            NotImplementedError: must be implemented by subclasses.
        """
        raise NotImplementedError

    @activity.setter
    @abstractmethod
    def activity(self, value: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def prediction(self, *pred: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Computes the single prediction for the node.

        Args:
            *pred (~torch.Tensor): predictions for the basis of initialization.
            **kwargs (~typing.Any): subclass-specific keyword arguments.

        Returns:
            ~torch.Tensor: prediction of the node's activity.

        Raises:
            NotImplementedError: must be implemented by subclasses.
        """
        raise NotImplementedError

    def initialize(self, *pred: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Initializes the node's state.

        Args:
            *pred (~torch.Tensor): predictions for the basis of initialization.
            **kwargs (~typing.Any): subclass-specific keyword arguments.

        Returns:
            ~torch.Tensor: the node's initial activity.
        """
        self.activity = self.prediction(*pred, **kwargs)
        return self.activity

    @abstractmethod
    def reset(self, **kwargs: Any) -> None:
        r"""Resets transient node state.

        Args:
            **kwargs (~typing.Any): subclass-specific keyword arguments.

        Raises:
            NotImplementedError: must be implemented by subclasses.
        """
        raise NotImplementedError

    def forward(self, *pred: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Computes a forward pass on the node.

        When ``self.training`` is True, the prediction is assigned to the value and then
        value is returned. When ``self.training`` is False, the prediction is directly
        returned (i.e. this acts as the identity operation).

        Args:
            *pred (~torch.Tensor): predictions of the node's activity.
            **kwargs (~typing.Any): subclass-specific keyword arguments.

        Returns:
            ~torch.Tensor: activity of the node.

        Note:
            Unless overridden by the subclass, keyword arguments are passed to
            ``self.initialize()`` when ``self.training`` is True, and to
            ``self.prediction()`` when ``self.training`` is False.
        """
        if self.training:
            return self.initialize(*pred, **kwargs)
        else:
            return self.prediction(*pred, **kwargs)


class PredictiveNode(Node, ABC):
    r"""Base class for predictive coding nodes with a defined energy function.

    Args:
        *shape (int | None): base shape of the node's state.
    """

    def __init__(self, *shape: int | None, **kwargs: Any) -> None:
        Node.__init__(self, *shape, **kwargs)

    @abstractmethod
    def energy(self, *pred: torch.Tensor) -> torch.Tensor:
        r"""Computes batchwise energy between predictions and the node's activity.

        Args:
            *pred (~torch.Tensor): predictions of the node's activity.
            **kwargs (~typing.Any): subclass-specific keyword arguments.

        Returns:
            ~torch.Tensor: batchwise energy between the activity and the predictions.

        Raises:
            NotImplementedError: must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def error(self, *pred: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Computes elementwise error between predictions and the node's activity.

        Args:
            *pred (~torch.Tensor): predictions of the node's activity.
            **kwargs (~typing.Any): subclass-specific keyword arguments.

        Returns:
            ~torch.Tensor: elementwise error between the activity and the predictions.

        Raises:
            NotImplementedError: must be implemented by subclasses.
        """
        raise NotImplementedError


class VariationalNode(PredictiveNode, ABC):
    r"""Base class for predictive coding nodes modelling a well-formed distribution.

    Args:
        *shape (int | None): base shape of the node's state.
    """

    def __init__(self, *shape: int | None, **kwargs: Any) -> None:
        PredictiveNode.__init__(self, *shape, **kwargs)

    def initialize(
        self,
        *pred: torch.Tensor,
        sample: bool = False,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Initializes the node's state.

        Args:
            *pred (~torch.Tensor): predictions for the basis of initialization.
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
        if sample:
            self.activity = self.sample(*pred, generator=generator, **kwargs)
        else:
            self.activity = self.prediction(*pred, **kwargs)

        return self.activity

    @abstractmethod
    def sample(
        self,
        *pred: torch.Tensor,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Samples from the learned conditional distribution.

        Args:
            *pred (~torch.Tensor): predictions of the node's activity, used as
                parameters of the distribution.
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
        *pred: torch.Tensor,
        sample: bool = False,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Computes a forward pass on the node.

        When ``self.training`` is True, the prediction is assigned to the value and then
        value is returned. When ``self.training`` is False, the prediction is directly
        returned (i.e. this acts as the identity operation).

        Args:
            *pred (~torch.Tensor): predictions of the node's activity.
            sample (bool, optional): if the activity should be initialized based on a
                sampled probability rather than the MAP estimate. Defaults to False.
            generator (~torch.Generator | None, optional): pseudorandom number generator
                for sampling. Defaults to None.
            **kwargs (~typing.Any): subclass-specific keyword arguments.

        Returns:
            ~torch.Tensor: activity of the node.

        Note:
            Unless overridden by the subclass, keyword arguments are passed to
            ``self.initialize()`` when ``self.training`` is True, and to either
            :py:meth:`self.prediction()` or ``self.sample()`` when ``self.training``
            is False.
        """
        if self.training:
            return self.initialize(*pred, sample=sample, generator=generator, **kwargs)
        elif sample:
            return self.sample(*pred, generator=generator, **kwargs)
        else:
            return self.prediction(*pred, **kwargs)


@eparameters("value")
class ValueNodeMixin:
    r"""Mixin for nodes where the activity is represented by a single tensor.

    Attributes:
        value (~torch.nn.parameter.Parameter): current value of the node.

    Important:
        In order for a class to inherit from this mixin, it must also inherit from
        :py:class:`Node`. Additionally, `Node.__init__()` must be called prior to
        `ValueNodeMixin.__init__()`.
    """

    value: nn.Parameter

    def __init__(self, **kwargs: Any) -> None:
        self.value = nn.Parameter(torch.empty(0), True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not issubclass(cls, Node):
            raise TypeError(
                f"{cls.__name__} must also inherit from Node to inherit "
                "from ValueNodeMixin"
            )

    @property
    def activity(self) -> nn.Parameter:
        r"""Activity of the node.

        Args:
            value (~torch.Tensor): value to set the activity to.

        Returns:
            ~torch.Tensor: activity of the node.
        """
        return self.value

    @activity.setter
    @torch.no_grad()
    def activity(self, value: torch.Tensor) -> None:
        if not self.shape.compat(*value.shape):  # type: ignore
            raise ValueError(
                f"shape of `value` {(*value.shape,)} is incompatible "
                f"with node shape {(*self.shape,)}"  # type: ignore
            )

        self.value.data = self.value.data.new_empty(*value.shape)
        self.value.copy_(value)

    @torch.no_grad()
    def reset(self, **kwargs) -> None:
        r"""Resets the node state.

        This operation is typically executed after each new batch. With inference learning,
        this is done after each M-step. With incremental inference learning, this is done
        after the *final* M-step.

        Args:
            **kwargs (~typing.Any): subclass-specific keyword arguments.
        """
        # assert isinstance(self, Node)
        self.zero_grad()  # type: ignore
        self.value.data = self.value.new_empty(0)
