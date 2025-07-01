from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from operator import itemgetter
from typing import Self

import einops as ein
import networkx as nx
import torch
import torch.nn as nn

from ..infra import LambdaModule, TypedModuleDict
from ..nodes import Node, PredictiveNode


class GraphSpec[T: Hashable]:
    r"""DiGraph with a total ordering on nodes.

    Args:
        graph (nx.DiGraph): underlying directed graph.
        order (Sequence[T]): ordering of nodes.

    Raises:
        RuntimeError: entries of ``order`` must be unique.
        RuntimeError: entries of ``order`` must be a exactly match ``graph.nodes``.

    Important:
        Type ``T`` must be a subtype of :py:type:`~typing.Hashable`.

    Note:
        Each node is treated as a symbol in a lexicographical order, and edges are
        sorted such that the parent node is the more significant symbol.
    """

    _graph: nx.DiGraph
    _order: dict[T, int]

    def __init__(self, graph: nx.DiGraph, order: Sequence[T]) -> None:
        self._graph = graph
        self._order = {node: pos for pos, node in enumerate(order)}

        if len(order) != len(self._order):
            raise RuntimeError("`order` cannot contain duplicate entries")
        if set(self._graph.nodes) != self._order.keys():
            raise RuntimeError(
                "`order` must contain the exactly the same nodes as `graph`"
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphSpec):
            return False
        if self._graph.nodes != other._graph.nodes:
            return False
        if self._graph.edges != other._graph.edges:
            return False
        return self._order == other._order

    @property
    def graph(self) -> nx.DiGraph:
        r"""Returns a resticted view of the underlying directed graph.

        Returns:
            nx.DiGraph: resticted view of the underlying directed graph.
        """
        return nx.restricted_view(self._graph, (), ())

    def reverse(self) -> Self:
        r"""Returns a new ``GraphSpec`` with the graph reversed and node order preserved.

        Returns:
            Self: new ``GraphSpec`` with the graph edges reversed.
        """
        return type(self)(self._graph.reverse(True), tuple(self._order.keys()))

    def nodes(self) -> list[T]:
        r"""Returns the graph nodes in sorted order.

        Returns:
            list[T]: graph nodes in sorted order.
        """
        return sorted(
            self._graph.nodes,
            key=lambda n, order=self._order: order[n],
        )

    def edges(self) -> list[tuple[T, T]]:
        r"""Returns the graph edges in sorted order.

        Returns:
            list[T]: graph edges in sorted order.
        """
        return sorted(
            self._graph.edges,
            key=lambda e, order=self._order: order[e[0]] * len(order) + order[e[1]],
        )

    def successors(self, node: T) -> list[T]:
        r"""Returns the successor nodes in sorted order.

        Args:
            node (T): node to get successors of.

        Returns:
            list[T]: successors of ``node`` in sorted order.
        """
        return sorted(
            self._graph.successors(node),
            key=lambda n, order=self._order: order[n],
        )

    def predecessors(self, node: T) -> list[T]:
        r"""Returns the predecessor nodes in sorted order.

        Args:
            node (T): node to get predecessors of.

        Returns:
            list[T]: predecessors of ``node`` in sorted order.
        """
        return sorted(
            self._graph.predecessors(node),
            key=lambda n, order=self._order: order[n],
        )

    def sort_nodes(self, nodes: Iterable[T]) -> list[T]:
        r"""Sort nodes into the specified ordering.

        Args:
            nodes (Iterable[T]): nodes to sort.

        Returns:
            list[T]: sorted nodes.
        """
        return sorted(
            nodes,
            key=lambda n, order=self._order: order[n],
        )

    def sort_edges(self, edges: Iterable[tuple[T, T]]) -> list[tuple[T, T]]:
        r"""Sort edges into the specified ordering.

        Args:
            edges (Iterable[tuple[T, T]]): edges to sort.

        Returns:
            list[tuple[T, T]]: sorted edges.
        """
        return sorted(
            edges,
            key=lambda e, order=self._order: order[e[0]] * len(order) + order[e[1]],
        )


class GraphNodeView:
    r"""Intractable view of a Node inside of a Graph.

    Args:
        node (Node): predictive coding node.
        join (~torch.nn.Module): join operation for inputs.
        predecessors (Sequence[tuple[Node, ~torch.nn.Module]]): tuples of
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
            Node: predictive coding node.
        """
        return self._node

    @property
    def prediction(self) -> torch.Tensor:
        r"""Returns the prediction for the value of the node.

        Returns:
            torch.Tensor: prediction for the value of the node.
        """
        return self._join([edge(node.activity) for node, edge in self._predecessors])

    @property
    def error(self) -> torch.Tensor:
        r"""Returns the error between the prediction and the value of the node.

        Returns:
            torch.Tensor: error between the prediction and the value of the node.
        """
        return self._node.error(self.prediction)

    @property
    def energy(self) -> torch.Tensor:
        r"""Returns the energy between the prediction and the value of the node.

        Raises:
            TypeError: only nodes of type :py:class:`~pyromancy.nodes.PredictionNode`
                support computing energy.

        Returns:
            torch.Tensor: energy between the prediction and the value of the node.
        """
        if not isinstance(self._node, PredictiveNode):
            raise TypeError("only `PredictionNode` nodes support `energy()`")
        return self._node.energy(self.prediction)


class Graph(nn.Module):
    r"""Predictive coding graph.

    Args:
        nodes (Mapping[str, Node]): nodes in the graph, mapped by a string identifier.
        edges (Mapping[tuple[str, str], nn.Module]): edges in the graph, representing
            connections between nodes, mapped by a tuple ``(source, target)``.
        joins (Mapping[str, Callable[[tuple[torch.Tensor, ...]], torch.Tensor]] | None, optional): method
            for joining multiple inputs for a node into a single prediction. Defaults to None.

    Raises:
        AttributeError: ``nodes`` cannot be empty.
        AttributeError: ``edges`` cannot be empty.
        TypeError: values in ``nodes`` must be a :py:class:`~pyromancy.nodes.Node`.
        TypeError: values in ``edges`` must be a :py:class:`~torch.nn.Module`.
        KeyError: key in ``edges`` cannot be used to specify a key in ``nodes``.
        KeyError: key in ``joins`` does not match a key in ``nodes``.
        RuntimeError: a join must be specified for any node with multiple inputs.
        TypeError: values in ``edges`` must be a :py:class:`~collections.abc.Callable`.
        RuntimeError: the graph cannot contain multiple disconnected subgraphs.
    """

    _spec: GraphSpec

    nodes: TypedModuleDict[Node]
    edges: TypedModuleDict[nn.Module]
    joins: TypedModuleDict[nn.Module]

    def __init__(
        self,
        nodes: Mapping[str, Node],
        edges: Mapping[tuple[str, str], nn.Module],
        joins: (
            Mapping[str, Callable[[tuple[torch.Tensor, ...]], torch.Tensor]] | None
        ) = None,
    ) -> None:
        if len(nodes) == 0:
            raise AttributeError("`nodes` cannot be empty")
        if len(edges) == 0:
            raise AttributeError("`edges` cannot be empty")

        nn.Module.__init__(self)

        _graph = nx.DiGraph()
        _order = []

        self.nodes = TypedModuleDict(narrowing=Node)
        self.edges = TypedModuleDict()
        self.joins = TypedModuleDict()

        # add nodes
        for name, node in nodes.items():
            if not isinstance(node, Node):
                raise TypeError(
                    f"item with key '{name}' in `nodes` is not a `Node` object"
                )

            self.nodes[name] = node
            _graph.add_node(name)
            _order.append(name)

        # add edges
        for pair, edge in edges.items():
            if not isinstance(edge, nn.Module):
                raise TypeError(
                    f"item with key {pair} in `edges` is not a `Module` object"
                )
            if not all(n in self.nodes for n in pair):
                raise KeyError(f"{pair} in `edges` specifies invalid an invalid `Node`")

            self.edges[f"{pair[0]} -> {pair[1]}"] = edge
            _graph.add_edge(pair[0], pair[1])

        # add joins
        for name, join in (
            {n: None for n in self.nodes} | dict(joins if joins else {})
        ).items():
            # ensure join is for a valid node
            if name not in self.nodes:
                raise KeyError(f"key '{name}' in `joins` is not specified in `nodes`")

            # create join automatically
            if join is None:
                # manual join specification required for multiple inputs
                if _graph.in_degree(name) > 1:  # type: ignore
                    raise RuntimeError(
                        f"`joins` must specify a join for node {name} "
                        f"with indegree {_graph.in_degree(name)}"
                    )

                # trivial retrieval for single input
                if _graph.in_degree(name) == 1:
                    self.joins[name] = LambdaModule(itemgetter(0))

                # identity placeholder for nodes without parents
                else:
                    self.joins[name] = nn.Identity()

            # ensure the join is a callable
            elif not isinstance(join, Callable):
                raise TypeError(
                    f"item with key '{name}' in `joins` is not a `Callable`"
                )

            # wrap with a LambdaModule if its not a module
            elif not isinstance(join, nn.Module):
                self.joins[name] = LambdaModule(join)

            # use directly if it is a module
            else:
                self.joins[name] = join

        # check for graph connectivity
        _ncomp = nx.number_weakly_connected_components(_graph)
        if _ncomp > 1:
            raise RuntimeError(
                f"graph can only have 1 weakly connected component, but it has {_ncomp}"
            )

        # create graph specification
        self._spec = GraphSpec(_graph, _order)

    @property
    def spec(self) -> GraphSpec:
        return self._spec

    def nodeview(self, node: str) -> GraphNodeView:
        return GraphNodeView(
            self.node(node),
            self.join(node),
            [
                (self.nodes(pred), self.edges(pred, node))
                for pred in self._spec.predecessors(node)
            ],
        )

    def node(self, node: str) -> Node:
        return self.nodes[node]  # type: ignore

    def edge(self, source: str, target: str) -> nn.Module:
        return self.edges[source + " -> " + target]

    def join(self, node: str) -> nn.Module:
        return self.joins[node]

    def reset(self) -> None:
        node: Node
        for node in self.nodes.values():  # type: ignore
            node.reset()

    def energy(self) -> torch.Tensor:
        energy = []

        # iterate over the "target" nodes
        for tgt, node in self.nodes.items():

            # get ordered predecessors
            predecessors = self._spec.predecessors(tgt)

            # skip nodes where energy cannot be computed
            if not isinstance(node, PredictiveNode) or not predecessors:
                continue

            # compute the energy from joint prediction
            pred = self.join(tgt)(
                [self.edge(src, tgt)(self.node(src).activity) for src in predecessors]
            )
            energy.append(node.energy(pred))

        # sum energy over nodes
        return ein.reduce(energy, "n ... -> ...", "sum")

    def init(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "`init()` is not directly callable on a `Graph`, create a `GraphExecutor`"
        )

    def forward(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "`forward()` is not directly callable on a `Graph`, create a `GraphExecutor`"
        )
