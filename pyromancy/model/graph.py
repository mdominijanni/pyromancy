from __future__ import annotations
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from operator import itemgetter

import einops as ein
import networkx as nx
import torch
import torch.nn as nn

from ..infra import LambdaModule, TypedModuleDict
from ..nodes import Node, PredictiveNode


class GraphSpec[T: Hashable]:
    r"""DiGraph with a total ordering on nodes.

    Args:
        graph (~networkx.DiGraph): underlying directed graph.
        node_order (Sequence[T]): ordering of nodes.
        edge_order (Sequence[tuple[T, T]]): ordering of edges.

    Raises:
        RuntimeError: ``graph`` must have exactly one weakly connected component.
        RuntimeError: entries of ``node_order`` must be unique.
        RuntimeError: entries of ``node_order`` must be a exactly match ``graph.nodes``.
        RuntimeError: entries of ``edge_order`` must be unique.
        RuntimeError: entries of ``edge_order`` must be a exactly match ``graph.edges``.

    Important:
        Type ``T`` must be a subtype of :py:type:`~typing.Hashable`.
    """

    _graph: nx.DiGraph
    _node_order: dict[T, int]
    _edge_order: dict[tuple[T, T], int]
    _pred_order: dict[T, dict[T, int]]
    _succ_order: dict[T, dict[T, int]]

    def __init__(
        self,
        graph: nx.DiGraph,
        node_order: Sequence[T],
        edge_order: Sequence[tuple[T, T]],
    ) -> None:
        # check that the entire graph is connected
        if nx.number_weakly_connected_components(graph) != 1:
            raise RuntimeError(
                "`graph` must have exactly one weakly connected component"
            )

        self._graph = graph
        self._node_order = {node: pos for pos, node in enumerate(node_order)}
        self._edge_order = {edge: pos for pos, edge in enumerate(edge_order)}

        if len(node_order) != len(self._node_order):
            raise RuntimeError("`node_order` cannot contain duplicate entries")
        if set(graph.nodes) != self._node_order.keys():
            raise RuntimeError(
                "`node_order` must contain the exactly the same nodes as `graph`"
            )

        if len(edge_order) != len(self._edge_order):
            raise RuntimeError("`edge_order` cannot contain duplicate entries")
        if set(graph.edges) != self._edge_order.keys():
            raise RuntimeError(
                "`edge_order` must contain the exactly the same edges as `graph`"
            )

        self._pred_order = {succ: {} for succ in self._node_order}
        self._succ_order = {pred: {} for pred in self._node_order}

        for pred, succ in self._edge_order:
            pred_pos = self._node_order[pred] * len(self._node_order)
            succ_pos = self._node_order[succ] * len(self._node_order)

            self._pred_order[succ][pred] = succ_pos + len(self._pred_order[succ])
            self._succ_order[pred][succ] = pred_pos + len(self._succ_order[pred])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphSpec):
            return False
        if self._graph.nodes != other._graph.nodes:
            return False
        if self._graph.edges != other._graph.edges:
            return False
        if self._node_order != other._node_order:
            return False
        if self._edge_order != other._edge_order:
            return False
        return True

    @property
    def graph(self) -> nx.DiGraph:
        r"""Returns a resticted view of the underlying directed graph.

        Returns:
            ~networkx.DiGraph: resticted view of the underlying directed graph.
        """
        return nx.restricted_view(self._graph, (), ())

    def copy(self) -> GraphSpec:
        r"""Returns a new ``GraphSpec`` that is a deep copy of the original.

        Returns:
            GraphSpec: deep copy of ``self``.
        """
        return GraphSpec(
            self._graph.copy(),
            tuple(self._node_order.keys()),
            tuple(self._edge_order.keys()),
        )

    def reverse(
        self, reverse_node_order: bool = False, reverse_edge_order: bool = False
    ) -> GraphSpec:
        r"""Returns a new ``GraphSpec`` with the graph reversed and node order preserved.

        Args:
            reverse_node_order (bool, optional): if the order of nodes should also
                be reversed. Defaults to ``False``.
            reverse_edge_order (bool, optional): if the order of edges should also
                be reversed. Defaults to ``False``.

        Returns:
            GraphSpec: new ``GraphSpec`` with the graph edges reversed.
        """
        node_order = tuple(self._node_order.keys())
        edge_order = tuple((succ, pred) for pred, succ in self._edge_order.keys())

        if reverse_node_order:
            node_order = tuple(reversed(node_order))
        if reverse_edge_order:
            edge_order = tuple(reversed(edge_order))

        return GraphSpec(self._graph.reverse(True), node_order, edge_order)

    def nodes(self) -> list[T]:
        r"""Returns the graph nodes in sorted order.

        Returns:
            list[T]: graph nodes in sorted order.
        """
        return sorted(
            self._graph.nodes,
            key=lambda n, order=self._node_order: order[n],
        )

    def edges(self) -> list[tuple[T, T]]:
        r"""Returns the graph edges in sorted order.

        Returns:
            list[T]: graph edges in sorted order.
        """
        return sorted(
            self._graph.edges,
            key=lambda e, order=self._edge_order: order[e],
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
            key=lambda n, order=self._succ_order[node]: order[n],
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
            key=lambda n, order=self._pred_order[node]: order[n],
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
            key=lambda n, order=self._node_order: order[n],
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
            key=lambda e, order=self._edge_order: order[e],
        )


class GraphNodeView:
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
        """
        return self._join([edge(node.activity) for node, edge in self._predecessors])

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
            raise TypeError("only `PredictionNode` nodes support `energy()`")
        return self._node.energy(self.prediction)


class Graph(nn.Module):
    r"""Predictive coding graph.

    Args:
        nodes (Mapping[str, ~pyromancy.nodes.Node]): nodes in the graph, mapped by a
            string identifier.
        edges (Mapping[tuple[str, str], ~torch.nn.Module]): edges in the graph, representing
            connections between nodes, mapped by a tuple ``(source, target)``.
        joins (Mapping[str, Callable[[tuple[~torch.Tensor, ...]], ~torch.Tensor]] | None, optional):
            method for joining multiple inputs for a node into a single prediction. Defaults to None.

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

    Tip:
        The values of the ``joins`` argument can be instances of :py:class:`~torch.nn.Module`
        and their E-step and M-step parameters will be retrieved from :py:class:`GraphExecutor`.
        Internally, any join that is not a :py:class:`~torch.nn.Module` will be wrapped
        with a :py:class:`LambdaModule`.

    Important:
        The tuple of tensors passed to each join are given in the same order as the
        edges with the corresponding target node are specified by the ``edges`` argument.
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
        _node_order = []
        _edge_order = []

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
            _node_order.append(name)

        # add edges
        for pair, edge in edges.items():
            if not isinstance(edge, nn.Module):
                raise TypeError(
                    f"item with key {pair} in `edges` is not a `Module` object"
                )
            if not all(n in self.nodes for n in pair):
                raise KeyError(f"{pair} in `edges` specifies invalid an invalid `Node`")

            self.edges[self.edgekey(*pair)] = edge
            _graph.add_edge(pair[0], pair[1])
            _edge_order.append(pair)

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
                        f"`joins` must specify a join for node '{name}' "
                        f"with indegree {_graph.in_degree(name)}"
                    )

                # trivial retrieval for single input
                # note: for nodes w/o parents, initialization value is put into a list
                else:
                    self.joins[name] = LambdaModule(itemgetter(0))

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

        # create graph specification
        self._spec = GraphSpec(_graph, _node_order, _edge_order)

    @staticmethod
    def edgekey(source: str, target: str) -> str:
        r"""Composes an edge key given a source and target node.

        Args:
            source (str): name of the source node.
            target (str): name of the target node.

        Returns:
            str: key specifying the edge from the source node to the target node.

        Important:
            This does not check the validity of ``source``, ``target``, or the existence
            of an edge between the two.
        """
        return source + " -> " + target

    @property
    def spec(self) -> GraphSpec:
        r"""Topology of the graph and the node ordering.

        Returns:
            GraphSpec: object containing the topology of the graph and the node ordering.
        """
        return self._spec

    def nodeview(self, node: str) -> GraphNodeView:
        r"""View of a node on which operations can be performed based on the graph.

        Args:
            node (str): name of the node to retrieve.

        Returns:
            GraphNodeView: view of a node on which operations can be performed based on the graph.
        """
        return GraphNodeView(
            self.node(node),
            self.join(node),
            [
                (self.nodes(pred), self.edges(pred, node))
                for pred in self._spec.predecessors(node)
            ],
        )

    def node(self, node: str) -> Node:
        r"""Returns a graph node.

        Args:
            node (str): name of the node.

        Returns:
            Node: graph node.
        """
        return self.nodes[node]

    def edge(self, source: str, target: str) -> nn.Module:
        r"""Returns a graph edge.

        Args:
            source (str): name of the node providing input to the edge.
            target (str): name of the node receiving output from the edge.

        Returns:
            nn.Module: graph edge between the source and target nodes.
        """
        return self.edges[self.edgekey(source, target)]

    def join(self, node: str) -> nn.Module:
        r"""Returns a graph join for a given node.

        Args:
            node (str): name of the node.

        Returns:
            nn.Module: graph join for the given node.
        """
        return self.joins[node]

    def reset(self) -> None:
        r"""Resets nodes in the network."""
        node: Node
        for node in self.nodes.values():
            node.reset()

    def energy(self) -> torch.Tensor:
        r"""Computes the variational free energy of the network.

        Returns:
            torch.Tensor: variational free energy of the network.

        Important:
            The output energy is not reduced along the batch dimension.

        Note:
            Nodes which are not instances of :py:class:`PredictiveNode` and therefore
            do not have a defined energy calculation are automatically excluded, as
            are nodes with no inputs.
        """
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
