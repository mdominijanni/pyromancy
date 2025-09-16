from __future__ import annotations
from collections.abc import Sequence
import torch
import torch.nn as nn

from ..infra import Shape
from ..nodes import Node, PredictiveNode


class BackpropNode(Node):
    r"""Placeholder node for use with backpropagation.

    Args:
        *shape (int | None): shape of the node's state.

    Tip:
        This does not support the normal inference learning paradigm of predictive coding.
        Instead, this can be used as a placeholder in :py:class:`~pyromancy.model.Graph`
        to test training with backpropagation.

    Important:
        When used in a :py:class:`~pyromancy.model.Graph`, every node with trainable
        E-step parameters should be instances of `BackpropNode`. Additionally,
        :py:class:`~pyromancy.model.GraphExecutor.init` should not be used. Instead,
        :py:class:`~pyromancy.model.GraphExecutor.forward` should be used, with loss
        computed on those outputs and backpropagated as normal.
    """

    value: torch.Tensor

    def __init__(self, *shape: int | None) -> None:
        Node.__init__(self, *shape)
        self.value = torch.empty(0)

    @property
    def activity(self) -> torch.Tensor:
        r"""Activity of the node.

        Returns:
            ~torch.Tensor: activity (state) of the node.
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
        self.value = self.value.new_empty(0)

    def init(self, value: torch.Tensor) -> torch.Tensor:
        r"""Initializes the node's state to a new value.

        Args:
            value (~torch.Tensor): value to initialize to.

        Returns:
            ~torch.Tensor: the reinitialized value.

        Raises:
            ValueError: shape of ``value`` is incompatible with the node.
        """
        if not self.shapeobj.compat(*value.shape):
            raise ValueError(
                f"shape of `value` {(*value.shape,)} is incompatible "
                f"with node shape {(*self.shapeobj,)}"
            )

        self.value = value
        return self.value

    def error_from(self, value: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        r"""Computes elementwise error for a prediction of the node state and its presumed state.

        .. math::
            \boldsymbol{\epsilon} = \mathbf{z} - \boldsymbol{\mu}

        Args:
            value (~torch.Tensor): presumed value of the node state :math:`\mathbf{z}`.
            pred (~torch.Tensor): predicted value :math:`\boldsymbol{\mu}`.

        Returns:
            ~torch.Tensor: elementwise error :math:`\boldsymbol{\epsilon}`.
        """
        return value - pred


class PlaceholderNode(Node):
    r"""Placeholder class for predictive coding nodes in graphs.

    Tip:
        These should never need to be directly constructed, but instead
        automatically filled by graphs.
    """

    def __init__(self) -> None:
        nn.Module.__init__(self)

    @property
    def shapeobj(self) -> Shape:
        r"""Object storing the node shape.

        Returns:
            Shape: object storing the node shape.

        Raise:
            TypeError: ``PlaceholderNode`` does not support standard
                :py:class:`~pyromancy.nodes.Node` methods.
        """
        raise TypeError("`PlaceholderNode` does not support `shapeobj` property")

    @property
    def shape(self) -> tuple[int | None, ...]:
        r"""Shape of the node state.

        Returns:
            tuple[int | None, ...]: shape of the node state.

        Note:
            Placeholder dimensions represented with ``None`` values.
            Use :py:meth:`~pyromancy.nodes.base.Node.bshape` for a version to use
            when constructing broadcastable tensors.

        Raise:
            TypeError: ``PlaceholderNode`` does not support standard
                :py:class:`~pyromancy.nodes.Node` methods.
        """
        raise TypeError("`PlaceholderNode` does not support `shape` property")

    @property
    def bshape(self) -> tuple[int, ...]:
        r"""Shape of the node state, safe for tensor construction.

        Returns:
            tuple[int, ...]: shape of the node state.

        Note:
            Placeholder dimensions represented with unit length dimensions.
            Use :py:meth:`~pyromancy.nodes.base.Node.shape` for a version to use
            that preserves placeholders.

        Raise:
            TypeError: ``PlaceholderNode`` does not support standard
                :py:class:`~pyromancy.nodes.Node` methods.
        """
        raise TypeError("`PlaceholderNode` does not support `bshape` property")

    @property
    def size(self) -> int:
        r"""Size of the node state.

        Returns:
            int: size of the node state.

        Raise:
            TypeError: ``PlaceholderNode`` does not support standard
                :py:class:`~pyromancy.nodes.Node` methods.
        """
        raise TypeError("`PlaceholderNode` does not support `size` property")

    @property
    def activity(self) -> torch.Tensor:
        r"""Activity of the node.

        Returns:
            ~torch.Tensor: activity (state) of the node.

        Raise:
            TypeError: ``PlaceholderNode`` does not support standard
                :py:class:`~pyromancy.nodes.Node` methods.
        """
        raise TypeError("`PlaceholderNode` does not support `size` property")

    def reset(self) -> None:
        r"""Resets transient node state.

        Raise:
            TypeError: ``PlaceholderNode`` does not support standard
                :py:class:`~pyromancy.nodes.Node` methods.
        """
        raise TypeError("`PlaceholderNode` does not support `reset` property")

    def init(self, value: torch.Tensor) -> torch.Tensor:
        r"""Initializes the node's state.

        Args:
            value (~torch.Tensor): value for the basis of initialization.

        Returns:
            ~torch.Tensorr: the reinitialized value.

        Raises:
            NotImplementedError: must be implemented by subclasses.
        """
        raise NotImplementedError

    def error_from(self, value: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        r"""Computes elementwise error for a prediction of the node state and its presumed state.

        Args:
            value (~torch.Tensor): presumed value of the node state.
            pred (~torch.Tensor): prediction of the node state.

        Returns:
            ~torch.Tensor: elementwise error between its presumed state and a prediction.

        Raises:
            NotImplementedError: must be implemented by subclasses.
        """
        raise NotImplementedError

    def error(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Computes elementwise error for a prediction of the node state.

        Args:
            pred (~torch.Tensor): prediction of the node state.

        Returns:
            ~torch.Tensor: elementwise error between the state and a prediction.
        """
        return self.error_from(self.activity, pred)

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


class NodeView:
    r"""Intractable view of a Node inside of a Graph.

    Args:
        node (~pyromancy.nodes.Node): predictive coding node.
        join (~torch.nn.Module): join operation for inputs.
        predecessors (Sequence[tuple[~pyromancy.nodes.Node, ~torch.nn.Module]]): tuples of
            ``(predecessor, edge)`` providing input to ``node``.

    Raises:
        TypeError: ``node`` must be of type :py:class:`~pyromancy.nodes.Node`.
        TypeError: ``join`` must be of type :py:class:`~torch.nn.Module`.
        TypeError: all elements of ``predecessors`` must be a ``tuple[Node, nn.Module]``.
    """

    _node: Node
    _join: nn.Module
    _predecessors: list[tuple[Node, nn.Module]]

    def __init__(
        self,
        node: Node,
        join: nn.Module,
        predecessors: Sequence[tuple[Node, nn.Module]],
    ) -> None:
        if not isinstance(node, Node):
            raise TypeError("`node` must be a `Node`")
        if not isinstance(join, nn.Module):
            raise TypeError("`join` must be an `nn.Module`")

        self._node = node
        self._join = join
        self._predecessors = []

        for pred, edge in predecessors:
            if not isinstance(pred, Node) or not isinstance(edge, nn.Module):
                raise TypeError(
                    "elements of `predecessors` must be a `tuple[Node, nn.Module]`"
                )
            self._predecessors.append((pred, edge))

    @property
    def node(self) -> Node:
        r"""Returns the predictive coding node.

        Returns:
            ~pyromancy.nodes.Node: predictive coding node.
        """
        return self._node

    @property
    def prediction(self) -> torch.Tensor:
        r"""Returns the prediction for the value of the node.

        Returns:
            ~torch.Tensor: prediction for the value of the node.

        Raises:
            RuntimeError: predictions can only be generated for nodes with predecessors.
        """
        if not self._predecessors:
            raise RuntimeError("cannot call `prediction` on a node without parents")
        return self._join(
            tuple(edge(node.activity) for node, edge in self._predecessors)
        )

    @property
    def error(self) -> torch.Tensor:
        r"""Returns the error between the prediction and the value of the node.

        Returns:
            ~torch.Tensor: error between the prediction and the value of the node.
        """
        return self._node.error(self.prediction)

    @property
    def energy(self) -> torch.Tensor:
        r"""Returns the energy between the prediction and the value of the node.

        Returns:
            ~torch.Tensor: energy between the prediction and the value of the node.

        Raises:
            TypeError: only nodes of type :py:class:`~pyromancy.nodes.PredictionNode`
                support computing energy.
        """
        if not isinstance(self._node, PredictiveNode):
            raise TypeError("only `PredictionNode` nodes support `energy`")
        return self._node.energy(self.prediction)
