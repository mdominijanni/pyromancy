import einops as ein
import networkx as nx
import pytest
import random
import torch
import torch.nn as nn

from .common import random_dag

from pyromancy.model import GraphSpec, GraphTrace, TraversalStrategy, ResolutionStrategy


class TestGraphTrace:

    @pytest.mark.parametrize(
        "resolver",
        (TraversalStrategy.BFS, TraversalStrategy.DFS),
        ids=("resolver=BFS", "resolver=DFS"),
    )
    def test_init_bad_ordertype(self, resolver):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C", "D"]
        edges = [("A", "B"), ("B", "C"), ("C", "D")]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        spec = GraphSpec(graph, nodes, edges)

        with pytest.raises(TypeError) as excinfo:
            _ = GraphTrace(spec, [[3]], resolver, skip_unreachable=False)  # type: ignore
        assert "nodes in `ordering` must be of type `str`" in str(excinfo.value)

    @pytest.mark.parametrize(
        "resolver",
        (TraversalStrategy.BFS, TraversalStrategy.DFS),
        ids=("resolver=BFS", "resolver=DFS"),
    )
    def test_init_bad_ordervalue(self, resolver):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C", "D"]
        edges = [("A", "B"), ("B", "C"), ("C", "D")]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        spec = GraphSpec(graph, nodes, edges)

        with pytest.raises(KeyError) as excinfo:
            _ = GraphTrace(spec, ["F"], resolver, skip_unreachable=False)
        assert "'F' is not a node in `spec.graph`" in str(excinfo.value)

    @pytest.mark.parametrize(
        "resolver",
        (TraversalStrategy.BFS, TraversalStrategy.DFS),
        ids=("resolver=BFS", "resolver=DFS"),
    )
    def test_init_bad_orderunderspec(self, resolver):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C", "D"]
        edges = [("A", "B"), ("B", "C"), ("C", "D")]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        spec = GraphSpec(graph, nodes, edges)

        with pytest.raises(RuntimeError) as excinfo:
            _ = GraphTrace(spec, ["C", "D"], resolver, skip_unreachable=True)
        assert "mismatch between nodes in `spec.graph` and nodes in `ordering`" in str(
            excinfo.value
        )

    @pytest.mark.parametrize(
        "resolver",
        (TraversalStrategy.BFS, TraversalStrategy.DFS),
        ids=("resolver=BFS", "resolver=DFS"),
    )
    def test_init_bad_unreachable(self, resolver):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C", "D"]
        edges = [("A", "B"), ("B", "C"), ("C", "D")]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        spec = GraphSpec(graph, nodes, edges)

        with pytest.raises(RuntimeError) as excinfo:
            _ = GraphTrace(
                spec, ["C", ["B", "A"], "D"], resolver, skip_unreachable=False
            )
        assert (
            "initializing nodes in `ordering[0]` cannot reach all nodes, missing: 'A', 'B'"
            in str(excinfo.value)
        )

    @pytest.mark.parametrize(
        "resolver",
        (TraversalStrategy.BFS, TraversalStrategy.DFS),
        ids=("resolver=BFS", "resolver=DFS"),
    )
    def test_initial_required_unknown(self, resolver):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C", "D", "E"]
        edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "D"), ("D", "E")]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        spec = GraphSpec(graph, nodes, edges)

        trace = GraphTrace(
            spec, [["B", "C"], "D", "E", "A"], resolver, skip_unreachable=True
        )

        assert set(trace.initial) == {"B", "C"}
        assert set(trace.required) == {"D"}
        assert set(trace.unknown) == {"A"}

    @pytest.mark.parametrize(
        "resolver",
        (TraversalStrategy.BFS, TraversalStrategy.DFS),
        ids=("resolver=BFS", "resolver=DFS"),
    )
    def test_process_t1(self, resolver):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C", "D", "E", "F", "G"]
        edges = [
            ("A", "B"),
            ("A", "F"),
            ("B", "C"),
            ("C", "D"),
            ("D", "D"),
            ("F", "E"),
            ("D", "E"),
            ("E", "F"),
            ("E", "G"),
        ]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        spec = GraphSpec(graph, nodes, edges)

        trace = GraphTrace(
            spec,
            [["C", "B"], "D", "E", "A", ["G", "F"]],
            resolver,
            skip_unreachable=True,
        )

        assert len(trace.process) == 4

        assert trace.process[0] == {
            "B": ((ResolutionStrategy.INITIAL, "B"),),
            "C": ((ResolutionStrategy.INITIAL, "C"),),
        }
        assert trace.process[1] == {
            "D": (
                (ResolutionStrategy.DERIVED, "C"),
                (ResolutionStrategy.HINTED, "D"),
            ),
        }
        assert trace.process[2] == {
            "E": (
                (ResolutionStrategy.HINTED, "F"),
                (ResolutionStrategy.DERIVED, "D"),
            ),
        }
        assert trace.process[3] == {
            "F": (
                (ResolutionStrategy.DERIVED, "A"),
                (ResolutionStrategy.DERIVED, "E"),
            ),
            "G": ((ResolutionStrategy.DERIVED, "E"),),
        }

    @pytest.mark.parametrize(
        "resolver",
        (TraversalStrategy.BFS, TraversalStrategy.DFS),
        ids=("resolver=BFS", "resolver=DFS"),
    )
    def test_process_t2(self, resolver):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C", "D", "E", "F", "G"]
        edges = [
            ("A", "B"),
            ("A", "F"),
            ("B", "C"),
            ("C", "D"),
            ("D", "D"),
            ("F", "E"),
            ("D", "E"),
            ("E", "F"),
            ("E", "G"),
        ]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        spec = GraphSpec(graph, nodes, edges)

        trace = GraphTrace(
            spec,
            [["C", "B"], "D", "E", ["G", "F"], "A"],
            resolver,
            skip_unreachable=True,
        )

        assert len(trace.process) == 4

        assert trace.process[0] == {
            "B": ((ResolutionStrategy.INITIAL, "B"),),
            "C": ((ResolutionStrategy.INITIAL, "C"),),
        }
        assert trace.process[1] == {
            "D": (
                (ResolutionStrategy.DERIVED, "C"),
                (ResolutionStrategy.HINTED, "D"),
            ),
        }
        assert trace.process[2] == {
            "E": (
                (ResolutionStrategy.HINTED, "F"),
                (ResolutionStrategy.DERIVED, "D"),
            ),
        }
        assert trace.process[3] == {
            "F": (
                (ResolutionStrategy.HINTED, "A"),
                (ResolutionStrategy.DERIVED, "E"),
            ),
            "G": ((ResolutionStrategy.DERIVED, "E"),),
        }

    @pytest.mark.parametrize(
        "resolver",
        (TraversalStrategy.BFS, TraversalStrategy.DFS),
        ids=("resolver=BFS", "resolver=DFS"),
    )
    def test_pathfind_dag_t1(self, resolver):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C", "D", "E", "F"]
        edges = [
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("C", "E"),
            ("E", "F"),
            ("D", "F"),
        ]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        spec = GraphSpec(graph, nodes, edges)

        trace = GraphTrace.pathfind_dag(spec, "A", resolver)

        assert len(trace.process) == 6

        assert trace.process[0] == {
            "A": ((ResolutionStrategy.INITIAL, "A"),),
        }
        assert trace.process[1] == {
            "B": ((ResolutionStrategy.DERIVED, "A"),),
        }
        assert trace.process[2] == {
            "C": ((ResolutionStrategy.DERIVED, "B"),),
        }
        assert trace.process[3] == {
            "D": ((ResolutionStrategy.DERIVED, "C"),),
        }
        assert trace.process[4] == {
            "E": ((ResolutionStrategy.DERIVED, "C"),),
        }
        assert trace.process[5] == {
            "F": (
                (ResolutionStrategy.DERIVED, "E"),
                (ResolutionStrategy.DERIVED, "D"),
            ),
        }

    @pytest.mark.parametrize(
        "resolver",
        (TraversalStrategy.BFS, TraversalStrategy.DFS),
        ids=("resolver=BFS", "resolver=DFS"),
    )
    def test_pathfind_dag_t2(self, resolver):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C", "D", "E", "F", "G", "H"]
        edges = [
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("C", "E"),
            ("D", "F"),
            ("E", "F"),
            ("F", "G"),
            ("G", "H"),
        ]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        spec = GraphSpec(graph, nodes, edges)

        trace = GraphTrace.pathfind_dag(spec, ["A", "G"], resolver)

        assert len(trace.process) == 7

        assert trace.process[0] == {
            "A": ((ResolutionStrategy.INITIAL, "A"),),
            "G": ((ResolutionStrategy.INITIAL, "G"),),
        }
        assert trace.process[1] == {
            "B": ((ResolutionStrategy.DERIVED, "A"),),
        }
        assert trace.process[2] == {
            "C": ((ResolutionStrategy.DERIVED, "B"),),
        }
        assert trace.process[3] == {
            "D": ((ResolutionStrategy.DERIVED, "C"),),
        }
        assert trace.process[4] == {
            "E": ((ResolutionStrategy.DERIVED, "C"),),
        }
        assert trace.process[5] == {
            "F": (
                (ResolutionStrategy.DERIVED, "D"),
                (ResolutionStrategy.DERIVED, "E"),
            ),
        }
        assert trace.process[6] == {
            "H": ((ResolutionStrategy.DERIVED, "G"),),
        }
