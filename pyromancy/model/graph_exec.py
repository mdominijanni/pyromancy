from __future__ import annotations
import itertools
from collections import ChainMap, deque
from collections.abc import Hashable, Iterator, KeysView, Sequence
from enum import Enum, auto

import networkx as nx
import torch
import torch.nn as nn

from .._internal import _Poset
from ..utils import (
    eparameters,
    get_named_estep_params,
    get_named_mstep_params,
    mparameters,
)
from .graph import Graph, GraphSpec


def _bfs_reachable_from[T: Hashable](
    graph: nx.DiGraph, nodes: Sequence[T]
) -> dict[T, None]:
    r"""Determines the reachable nodes in a directed graph, traversed breadth-first.

    Args:
        graph (nx.DiGraph): directed graph to search in.
        nodes (Sequence[T]): nodes to use as the search's starting point.

    Returns:
        dict[T, None]: nodes in ``graph`` reachable from ``nodes``.

    Important:
        Type ``T`` must be a subtype of :py:type:`~typing.Hashable`.
    """
    reachable = {}
    frontier: deque[T] = deque({n: None for n in nodes})

    while frontier:
        # get next node from the frontier
        node = frontier.popleft()

        # skip if we've visited the node
        if node in reachable:
            continue

        # mark the node as reachable and add its children to the frontier
        reachable[node] = None
        for succ in graph.successors(node):
            if succ not in reachable:
                frontier.append(succ)

    return reachable


def _dfs_reachable_from[T: Hashable](
    graph: nx.DiGraph, nodes: Sequence[T]
) -> dict[T, None]:
    r"""Determines the reachable nodes in a directed graph, traversed depth-first.

    Args:
        graph (nx.DiGraph): directed graph to search in.
        nodes (Sequence[T]): nodes to use as the search's starting point.

    Returns:
        dict[T, None]: nodes in ``graph`` reachable from ``nodes``.

    Important:
        Type ``T`` must be a subtype of :py:type:`~typing.Hashable`.
    """
    reachable = {}
    frontier: deque[T] = deque({n: None for n in nodes})

    while frontier:
        # get next node from the frontier
        node = frontier.popleft()

        # skip if we've visited the node
        if node in reachable:
            continue

        # mark the node as reachable and add its children to the frontier
        reachable[node] = None
        for succ in graph.successors(node):
            if succ not in reachable:
                frontier.appendleft(succ)

    return reachable


class TraversalStrategy(Enum):
    r"""Strategies for graph traversal used when tracing.

    - ``BFS``: breadth-first traversal.
    - ``DFS``: depth-first traversal.
    """

    BFS = auto()
    DFS = auto()


class ResolutionStrategy(Enum):
    r"""Strategies for resolving nodes in a trace."""

    INITIAL = auto()
    HINTED = auto()
    DERIVED = auto()

    def display(self, node: str) -> str:
        r"""Returns a string representation of a node name with resolution strategy.

        Args:
            node (str): name of the node.

        Returns:
            str: string with the node name and the resolution strategy.
        """
        match self:
            case self.__class__.INITIAL:
                return f"Initial({node})"
            case self.__class__.HINTED:
                return f"Hinted({node})"
            case self.__class__.DERIVED:
                return f"Derived({node})"


class GraphTrace:
    r"""Trace of an execution sequence for a digraph from initial nodes.

    Args:
        spec (GraphSpec): object containing the digraph and node order.
        ordering (Sequence[str | Sequence[str]]): order in which nodes are resolved, where
            ``ordering[0]`` are initializing nodes.
        traversal_strategy (TraversalStrategy, optional): method by which the graph should
            be traversed when testing reachability. Defaults to ``TraversalStrategy.BFS.``
        skip_unreachable (bool, optional): if unreachable nodes should be excluded from
            the resolution. Defaults to ``False``.

    Attributes:
        process (list[dict[str, tuple[tuple[ResolutionStrategy, str], ...]]]): list of stages,
            where each stage has a dictionary with the node to resolve as the key, and a tuple
            of the required node inputs and how to resolve them.

    Raises:
        TypeError: nodes in ``ordering`` must be of type :py:class:`str`.
        KeyError: a node in ``ordering`` is not in ``spec.graph.nodes``.
        TypeError: ``traversal_strategy`` is not a valid :py:class:`TraversalStrategy`.
        RuntimeError: trace from initializing nodes don't reach all other nodes (only when ``skip_unreachable=False``).
        RuntimeError: ``ordering`` must contain the exact same nodes as ``spec.graph.nodes``.
        RuntimeError: resolution for a node could not be computed (occurs for unreachable nodes).
    """

    process: list[dict[str, tuple[tuple[ResolutionStrategy, str], ...]]]
    _spec: GraphSpec
    _initial: dict[str, None]
    _required: dict[str, None]
    _unknown: dict[str, None]

    def __init__(
        self,
        spec: GraphSpec,
        ordering: Sequence[str | Sequence[str]],
        traversal_strategy: TraversalStrategy = TraversalStrategy.BFS,
        skip_unreachable: bool = False,
    ) -> None:
        # copy the graph spec
        self._spec = spec.copy()

        # get a view of the graph
        graph = self._spec.graph

        # repackage ordering
        ordering = [
            [*group] if not isinstance(group, str) else [group] for group in ordering
        ]

        # validate ordering
        for group in ordering:
            for node in group:
                if not isinstance(node, str):
                    raise TypeError("nodes in `ordering` must be of type `str`")
                if not graph.has_node(node):
                    raise KeyError(f"'{node}' is not a node in `spec.graph`")

        match traversal_strategy:
            case TraversalStrategy.BFS:
                reach = _bfs_reachable_from(graph, ordering[0])
            case TraversalStrategy.DFS:
                reach = _dfs_reachable_from(graph, ordering[0])
            case _:
                raise TypeError(
                    "`traversal_strategy` is not a valid `TraversalStrategy"
                )

        if not skip_unreachable and set(graph.nodes) != reach.keys():
            raise RuntimeError(
                f"initializing nodes in `ordering[0]` cannot reach all nodes, missing: "
                f"{', '.join(n for n in reach if n not in graph.nodes)}"
            )

        # define search variables
        unexplored = _Poset(ordering)
        if set(graph.nodes) != unexplored.keys():
            raise RuntimeError(
                "mismatch between nodes in `spec.graph` and nodes in `ordering`"
            )

        self.process = []
        self._initial = {}
        self._required = {}
        self._unknown = {}
        pruned: nx.DiGraph = graph.copy()  # type: ignore

        # process initializing nodes
        self.process.append(
            {n: ((ResolutionStrategy.INITIAL, n),) for n in unexplored.rank(0)}
        )
        self._initial |= {n: None for n in unexplored.rank(0)}
        pruned.remove_nodes_from(unexplored.rank(0))

        # explore the graph
        for rank in range(1, unexplored.nranks):
            group = {}

            # resolve nodes
            for node in self._spec.sort_nodes(unexplored.rank(rank)):
                # skip unreachable
                if skip_unreachable and node not in reach:
                    self._unknown[node] = None
                    continue

                # build resolution
                resolution = []

                # check all dependencies of the node
                for dep in self._spec.predecessors(node):
                    # unresolved dependency
                    if pruned.has_predecessor(node, dep):
                        resolution.append((ResolutionStrategy.HINTED, dep))
                        self._required[dep] = None
                    # resolved dependency
                    else:
                        resolution.append((ResolutionStrategy.DERIVED, dep))

                # check that the node has parents
                # note: this should be prevented so long as the nodes are reachable
                if not resolution:
                    raise RuntimeError(
                        f"a valid resolution for node '{node}' could not be found, "
                        "try setting it as an initial node"
                    )

                # add node resolution to working group
                group[node] = tuple(resolution)

            # add the group to the explored nodes
            if not len(group) == 0:
                self.process.append(group)

            # mark the nodes as resolved
            pruned.remove_nodes_from(unexplored.rank(rank))

    @classmethod
    def pathfind_dag(
        cls,
        spec: GraphSpec,
        initial: Sequence[str] | str,
        traversal_strategy: TraversalStrategy = TraversalStrategy.BFS,
        skip_unreachable: bool = False,
    ) -> GraphTrace:
        r"""Creates a trace for a directed acyclic graph using only the initial nodes.

        Args:
            spec (GraphSpec): object containing the DAG and node order.
            initial (Sequence[str] | str): nodes to use as the starting point of the trace.
            traversal_strategy (TraversalStrategy, optional): method by which the graph should
                be traversed when testing reachability. Defaults to ``TraversalStrategy.BFS.``
            skip_unreachable (bool, optional): if unreachable nodes should be excluded from
                the resolution. Defaults to ``False``.

        Returns:
            GraphTrace: trace of the graph.

        Raises:
            KeyError: all nodes in ``initial`` must also be in ``spec.graph``.
            RuntimeError: topologically sorting ``spec.graph`` failed, only DAGs
                can be topologically sorted.
        """
        # get immutable view of the graph
        graph = spec.graph

        # wrap initial in a sequence if its not
        if isinstance(initial, str):
            initial = (initial,)

        # check that nodes are valid
        for node in initial:
            if not graph.has_node(node):
                raise KeyError(f"`spec.graph` has no node {node}")

        # attempt to perform a topological sort
        try:
            ordering = {k: None for k in nx.topological_sort(graph)}
        except nx.NetworkXUnfeasible:
            raise RuntimeError(
                "could not topologically sort `spec.graph`, ensure it is a DAG"
            )

        # delete initial nodes from topological ordering
        for node in set(initial):
            del ordering[node]

        # create a trace with a manual path
        return cls(
            spec,
            [initial, *ordering.keys()],
            traversal_strategy=traversal_strategy,
            skip_unreachable=skip_unreachable,
        )

    def __repr__(self) -> str:
        indent = "  "
        dpad = len(str(len(self.process)))
        npad = max(len(n) for g in self.process for n in g)

        disp = [f"{type(self).__name__}("]

        for depth, group in enumerate(self.process):
            disp.append(f"{indent}{str(depth).rjust(dpad)}:")
            for node, deps in group.items():
                disp.append(
                    f"{indent * 2}{f'{node}:'.ljust(npad + 1)} "
                    f"{', '.join(d.display(n) for d, n in deps)}"
                )

        disp.append(")")
        return "\n".join(disp)

    @property
    def spec(self) -> GraphSpec:
        r"""Specification of the graph for the trace to be valid

        Returns:
            GraphSpec[str]: specification of the graph for the trace to be valid.
        """
        return self._spec

    @property
    def initial(self) -> KeysView[str]:
        r"""Nodes requiring manual initialization.

        Returns:
            KeysView[str]: nodes requiring manual initialization.
        """
        return self._initial.keys()

    @property
    def required(self) -> KeysView[str]:
        r"""Nodes requiring a hint during initializaiton.

        Returns:
            KeysView[str]: nodes requiring a hint during initializaiton.
        """
        return self._required.keys()

    @property
    def unknown(self) -> KeysView[str]:
        r"""Nodes unreachable by the initialization trace.

        Returns:
            KeysView[str]: nodes unreachable by the initialization trace.
        """
        return self._unknown.keys()


@eparameters()
@mparameters()
class GraphExecutor(nn.Module):
    r"""Provides functionality to execute predictive coding graphs

    Args:
        graph (Graph): graph over which to execute.
        trace (GraphTrace): initialization path to use when executing.

    Attributes:
        graph (Graph): the predictive coding graph being executed.

    Raises:
        RuntimeError: ``graph`` and ``trace`` have incompatible
            :py:class:`~pyromancy.model.GraphSpec` objects.
    """

    graph: Graph
    _trace: GraphTrace

    def __init__(self, graph: Graph, trace: GraphTrace) -> None:
        if graph.spec != trace.spec:
            raise RuntimeError("provided `graph` and `trace` are incompatible")

        nn.Module.__init__(self)

        self._trace = trace

    @property
    def required_inits(self) -> KeysView[str]:
        return self._trace.initial

    @property
    def required_hints(self) -> KeysView[str]:
        return self._trace.required

    def named_estep_params(
        self,
        exclude_initial: bool = True,
        manual_exclude: Sequence[nn.Parameter | nn.Module] | None = None,
        remove_duplicate=True,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        # set manual exclusions
        if manual_exclude is None:
            exclude = []
        else:
            exclude = [*manual_exclude]

        # add initial exclusions
        if exclude_initial:
            g = self.graph.spec.graph
            exclude += [
                self.graph.node(node)
                for node in g.nodes
                if node not in self._trace.initial
            ]
            exclude += [
                self.graph.edge(src, tgt)
                for src, tgt in g.edges
                if src not in self._trace.initial and tgt not in self._trace.initial
            ]
            exclude += [
                self.graph.join(node)
                for node in g.nodes
                if node not in self._trace.initial
            ]

        return iter(
            itertools.chain(
                get_named_estep_params(
                    self.graph.nodes,
                    default=False,
                    exclude=exclude,
                    prefix="graph.nodes",
                    remove_duplicate=remove_duplicate,
                ),
                get_named_estep_params(
                    self.graph.edges,
                    default=False,
                    exclude=exclude,
                    prefix="graph.edges",
                    remove_duplicate=remove_duplicate,
                ),
                get_named_estep_params(
                    self.graph.joins,
                    default=False,
                    exclude=exclude,
                    prefix="graph.joins",
                    remove_duplicate=remove_duplicate,
                ),
            )
        )

    def estep_params(
        self,
        exclude_initial: bool = True,
        manual_exclude: Sequence[nn.Parameter | nn.Module] | None = None,
    ) -> Iterator[nn.Parameter]:
        for _, p in self.named_estep_params(exclude_initial, manual_exclude, True):
            yield p

    def named_mstep_params(
        self,
        exclude_initial: bool = False,
        manual_exclude: Sequence[nn.Parameter | nn.Module] | None = None,
        remove_duplicate=True,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        # set manual exclusions
        if manual_exclude is None:
            exclude = []
        else:
            exclude = [*manual_exclude]

        # add initial exclusions
        if exclude_initial:
            g = self.graph.spec.graph
            exclude += [
                self.graph.node(node)
                for node in g.nodes
                if node not in self._trace.initial
            ]
            exclude += [
                self.graph.edge(src, tgt)
                for src, tgt in g.edges
                if src not in self._trace.initial and tgt not in self._trace.initial
            ]
            exclude += [
                self.graph.join(node)
                for node in g.nodes
                if node not in self._trace.initial
            ]

        return iter(
            itertools.chain(
                get_named_mstep_params(
                    self.graph.nodes,
                    default=True,
                    exclude=exclude,
                    prefix="graph.nodes",
                    remove_duplicate=remove_duplicate,
                ),
                get_named_mstep_params(
                    self.graph.edges,
                    default=True,
                    exclude=exclude,
                    prefix="graph.edges",
                    remove_duplicate=remove_duplicate,
                ),
                get_named_mstep_params(
                    self.graph.nodes,
                    default=True,
                    exclude=exclude,
                    prefix="graph.joins",
                    remove_duplicate=remove_duplicate,
                ),
            )
        )

    def mstep_params(
        self,
        exclude_initial: bool = False,
        manual_exclude: Sequence[nn.Parameter | nn.Module] | None = None,
    ) -> Iterator[nn.Parameter]:
        for _, p in self.named_mstep_params(exclude_initial, manual_exclude, True):
            yield p

    def reset(self) -> None:
        self.graph.reset()

    def init(
        self, initial: dict[str, torch.Tensor], hints: dict[str, torch.Tensor]
    ) -> None:
        for target, ops in ChainMap(*self._trace.process).items():
            inputs = []
            for res, source in ops:
                match res:
                    case ResolutionStrategy.INITIAL:
                        inputs.append(initial[source])
                    case ResolutionStrategy.HINTED:
                        inputs.append(self.graph.edge(source, target)(hints[source]))
                    case ResolutionStrategy.DERIVED:
                        inputs.append(
                            self.graph.edge(source, target)(
                                self.graph.node(source).activity
                            )
                        )
                    case _:
                        raise RuntimeError(
                            "internal trace contains an invalid ResolutionStrategy"
                        )
                self.graph.node(target).init(self.graph.join(target)(inputs))

    def energy(self) -> torch.Tensor:
        return self.graph.energy()

    def forward(
        self, initial: dict[str, torch.Tensor], hints: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        output = {}
        for target, ops in ChainMap(*self._trace.process).items():
            inputs = []
            for res, source in ops:
                match res:
                    case ResolutionStrategy.INITIAL:
                        inputs.append(initial[source])
                    case ResolutionStrategy.HINTED:
                        inputs.append(self.graph.edge(source, target)(hints[source]))
                    case ResolutionStrategy.DERIVED:
                        inputs.append(self.graph.edge(source, target)(output[source]))
                    case _:
                        raise RuntimeError(
                            "internal trace contains an invalid ResolutionStrategy"
                        )
                output[target] = self.graph.node(target)(
                    self.graph.join(target)(inputs)
                )

        return output
