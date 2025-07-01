from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from operator import itemgetter
from typing import Self

import einops as ein
import networkx as nx
import torch
import torch.nn as nn

from ..infra import LambdaModule
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

    nodes: nn.ModuleDict
    edges: nn.ModuleDict
    joins: nn.ModuleDict

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

        self.nodes = nn.ModuleDict()
        self.edges = nn.ModuleDict()
        self.joins = nn.ModuleDict()

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

            self.edges[f"{pair[0]}, {pair[1]}"] = edge
            _graph.add_edge(pair[0], pair[1])

        # add joins
        for name, join in (
            {n: None for n in self.nodes} | dict(joins if joins else {})
        ).items():
            # ensure join is for a valid node
            if name not in self.nodes:
                raise KeyError(f"key '{name}' in `joins` is not specified in `nodes`")

            if join is None:
                if _graph.in_degree(name) > 1:  # type: ignore
                    raise RuntimeError(
                        f"`joins` must specify a join for node {name} "
                        f"with indegree {_graph.in_degree(name)}"
                    )
                self.joins[name] = LambdaModule(itemgetter(0))
            elif not isinstance(join, Callable):
                raise TypeError(
                    f"item with key '{name}' in `joins` is not a `Callable`"
                )
            elif not isinstance(join, nn.Module):
                self.joins[name] = LambdaModule(join)
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
    def topology(self) -> nx.DiGraph:
        return self._spec.graph

    def node(self, node: str) -> Node:
        return self.nodes[node]  # type: ignore

    def edge(self, source: str, target: str) -> nn.Module:
        return self.edges[source + ", " + target]

    def join(self, node: str) -> nn.Module:
        return self.joins[node]

    def reset(self) -> None:
        node: Node
        for node in self.nodes.values():  # type: ignore
            node.reset()

    def predof(self, node: str) -> torch.Tensor:
        inputs = []
        for source in self._spec.predecessors(node):
            inputs.append(self.edge(source, node)(self.nodes[source].activity))
        return self.join(node)(inputs)

    def errorof(self, node: str) -> torch.Tensor:
        return self.node(node).error(self.predof(node))

    def energyof(self, node: str) -> torch.Tensor:
        nodeobj = self.node(node)
        if isinstance(nodeobj, PredictiveNode):
            return nodeobj.energy(self.predof(node))
        else:
            raise TypeError(
                "energy can only be computed on nodes of type `PredictiveNode`"
            )

    def energy(self) -> torch.Tensor:
        return ein.reduce(
            [
                self.energyof(node)
                for node in self.nodes
                if isinstance(node, PredictiveNode)
            ],
            "n ... -> ...",
            "sum",
        )

    def init(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "`init()` is not directly callable on a `Graph`, create a `GraphExecutor`"
        )

    def forward(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "`forward()` is not directly callable on a `Graph`, create a `GraphExecutor`"
        )
