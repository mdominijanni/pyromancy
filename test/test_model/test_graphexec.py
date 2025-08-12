import networkx as nx
import pytest
import random
import torch
import torch.nn as nn

from pyromancy import (
    eparameters,
    mparameters,
    get_named_estep_params,
    get_named_mstep_params,
)
from pyromancy.nodes import FactorizedGaussianNode

from .common import random_dag

from pyromancy.model import (
    Graph,
    GraphSpec,
    GraphTrace,
    TraversalStrategy,
    ResolutionStrategy,
    GraphExecutor,
)


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


class TestGraphExecutor:

    def test_required_inits(self):
        valid = False
        for _ in range(100):
            dag = random_dag([f"n{n}" for n in range(9)], 0.2)

            _nodes = [*dag.nodes]
            _edges = [*dag.edges]

            random.shuffle(_nodes)
            random.shuffle(_edges)

            spec = GraphSpec(dag, _nodes, _edges)
            try:
                trace = GraphTrace.pathfind_dag(spec, ["n0", "n8"])
            except RuntimeError:
                pass
            else:
                valid = True

            if valid:
                break
        assert valid

        def jsum(data: tuple[torch.Tensor, ...]) -> torch.Tensor:
            if not data:
                res = torch.empty(0)
            else:
                res = data[0]
                for d in data[1:]:
                    res = res + d

            return res

        g = Graph(
            nodes={n: FactorizedGaussianNode(10, variance=1.2) for n in _nodes},  # type: ignore
            edges={e: nn.Linear(10, 10) for e in _edges},  # type: ignore
            joins={n: jsum for n in _nodes if dag.in_degree(n) > 1},  # type: ignore
        )
        gx = GraphExecutor(g, trace)  # type: ignore

        assert set(gx.required_inits) == {"n0", "n8"}

    def test_required_hints(self):
        _nodes = ["n0", "n1", "n2", "n3"]
        _edges = [("n0", "n1"), ("n1", "n1"), ("n1", "n2"), ("n2", "n3")]

        def jsum(data: tuple[torch.Tensor, ...]) -> torch.Tensor:
            if not data:
                res = torch.empty(0)
            else:
                res = data[0]
                for d in data[1:]:
                    res = res + d

            return res

        g = Graph(
            nodes={n: FactorizedGaussianNode(10, variance=1.2) for n in _nodes},
            edges={e: nn.Linear(10, 10) for e in _edges},
            joins={"n1": jsum},
        )
        trace = GraphTrace(g.spec, ["n0", "n1", "n2", "n3"])
        gx = GraphExecutor(g, trace)

        assert set(gx.required_hints) == {"n1"}

    def test_init_dag(self):
        valid = False
        for _ in range(100):
            dag = random_dag([f"n{n}" for n in range(9)], 0.2)

            _nodes = [*dag.nodes]
            _edges = [*dag.edges]

            random.shuffle(_nodes)
            random.shuffle(_edges)

            spec = GraphSpec(dag, _nodes, _edges)
            try:
                trace = GraphTrace.pathfind_dag(spec, ["n0", "n8"])
            except RuntimeError:
                pass
            else:
                valid = True

            if valid:
                break
        assert valid

        def jsum(data: tuple[torch.Tensor, ...]) -> torch.Tensor:
            if not data:
                res = torch.empty(0)
            else:
                res = data[0]
                for d in data[1:]:
                    res = res + d

            return res

        g = Graph(
            nodes={n: FactorizedGaussianNode(10, variance=1.2) for n in _nodes},  # type: ignore
            edges={e: nn.Linear(10, 10) for e in _edges},  # type: ignore
            joins={n: jsum for n in _nodes if dag.in_degree(n) > 1},  # type: ignore
        )
        gx = GraphExecutor(g, trace)  # type: ignore
        inits = {"n0": torch.rand(5, 10), "n8": torch.rand(5, 10)}
        gx.init(inits)

        assert torch.all(g.node("n0").activity == inits["n0"])
        assert torch.all(g.node("n8").activity == inits["n8"])

        for n in _nodes:  # type: ignore
            if n in inits:
                continue
            nv = g.nodeview(n)
            assert torch.all(nv.node.activity == nv.prediction)

    def test_init_cyc(self):
        _nodes = ["n0", "n1", "n2", "n3"]
        _edges = [("n0", "n1"), ("n1", "n1"), ("n1", "n2"), ("n2", "n3")]

        def jsum(data: tuple[torch.Tensor, ...]) -> torch.Tensor:
            if not data:
                res = torch.empty(0)
            else:
                res = data[0]
                for d in data[1:]:
                    res = res + d

            return res

        g = Graph(
            nodes={n: FactorizedGaussianNode(10, variance=1.2) for n in _nodes},
            edges={e: nn.Linear(10, 10) for e in _edges},
            joins={"n1": jsum},
        )
        trace = GraphTrace(g.spec, ["n0", "n1", "n2", "n3"])
        gx = GraphExecutor(g, trace)
        inits = {"n0": torch.rand(5, 10)}
        hints = {"n1": torch.rand(5, 10)}
        gx.init(inits, hints)

        assert torch.all(g.node("n0").activity == inits["n0"])
        assert torch.all(
            g.node("n1").activity
            == g.edge("n0", "n1")(g.node("n0").activity)
            + g.edge("n1", "n1")(hints["n1"])
        )
        assert torch.all(
            g.node("n2").activity == g.edge("n1", "n2")(g.node("n1").activity)
        )
        assert torch.all(
            g.node("n3").activity == g.edge("n2", "n3")(g.node("n2").activity)
        )

    def test_energy(self):
        valid = False
        for _ in range(100):
            dag = random_dag([f"n{n}" for n in range(9)], 0.2)

            _nodes = [*dag.nodes]
            _edges = [*dag.edges]

            random.shuffle(_nodes)
            random.shuffle(_edges)

            spec = GraphSpec(dag, _nodes, _edges)
            try:
                trace = GraphTrace.pathfind_dag(spec, ["n0", "n8"])
            except RuntimeError:
                pass
            else:
                valid = True

            if valid:
                break
        assert valid

        def jsum(data: tuple[torch.Tensor, ...]) -> torch.Tensor:
            if not data:
                res = torch.empty(0)
            else:
                res = data[0]
                for d in data[1:]:
                    res = res + d

            return res

        g = Graph(
            nodes={n: FactorizedGaussianNode(10, variance=1.2) for n in _nodes},  # type: ignore
            edges={e: nn.Linear(10, 10) for e in _edges},  # type: ignore
            joins={n: jsum for n in _nodes if dag.in_degree(n) > 1},  # type: ignore
        )
        gx = GraphExecutor(g, trace)  # type: ignore
        inits = {"n0": torch.rand(5, 10), "n8": torch.rand(5, 10)}
        gx.init(inits)

        assert torch.allclose(gx.energy(), g.energy())

    @pytest.mark.parametrize(
        "training",
        (True, False),
        ids=("training=True", "training=False"),
    )
    def test_forward_dag(self, training):
        valid = False
        for _ in range(100):
            dag = random_dag([f"n{n}" for n in range(9)], 0.2)

            _nodes = [*dag.nodes]
            _edges = [*dag.edges]

            random.shuffle(_nodes)
            random.shuffle(_edges)

            spec = GraphSpec(dag, _nodes, _edges)
            try:
                trace = GraphTrace.pathfind_dag(spec, ["n0", "n8"])
            except RuntimeError:
                pass
            else:
                valid = True

            if valid:
                break
        assert valid

        def jsum(data: tuple[torch.Tensor, ...]) -> torch.Tensor:
            if not data:
                res = torch.empty(0)
            else:
                res = data[0]
                for d in data[1:]:
                    res = res + d

            return res

        g = Graph(
            nodes={n: FactorizedGaussianNode(10, variance=1.2) for n in _nodes},  # type: ignore
            edges={e: nn.Linear(10, 10) for e in _edges},  # type: ignore
            joins={n: jsum for n in _nodes if dag.in_degree(n) > 1},  # type: ignore
        )
        gx = GraphExecutor(g, trace)  # type: ignore
        gx.train(training)
        inits = {"n0": torch.rand(5, 10), "n8": torch.rand(5, 10)}

        res = gx(inits)

        assert torch.all(res["n0"] == inits["n0"])
        assert torch.all(res["n8"] == inits["n8"])

        if training:
            assert torch.all(g.node("n0").activity == inits["n0"])
            assert torch.all(g.node("n8").activity == inits["n8"])
        else:
            assert g.node("n0").activity.numel() == 0
            assert g.node("n8").activity.numel() == 0

        for n in _nodes:  # type: ignore
            if n in inits:
                continue

            if not training:
                assert g.node(n).activity.numel() == 0

        if not training:
            for n in _nodes:  # type: ignore
                g.node(n).init(res[n])

        for n in _nodes:  # type: ignore
            if n in inits:
                continue
            nv = g.nodeview(n)

            assert torch.all(res[n] == nv.prediction)

            if training:
                assert torch.all(nv.node.activity == nv.prediction)

    @pytest.mark.parametrize(
        "training",
        (True, False),
        ids=("training=True", "training=False"),
    )
    def test_forward_cyc(self, training):
        _nodes = ["n0", "n1", "n2", "n3"]
        _edges = [("n0", "n1"), ("n1", "n1"), ("n1", "n2"), ("n2", "n3")]

        def jsum(data: tuple[torch.Tensor, ...]) -> torch.Tensor:
            if not data:
                res = torch.empty(0)
            else:
                res = data[0]
                for d in data[1:]:
                    res = res + d

            return res

        g = Graph(
            nodes={n: FactorizedGaussianNode(10, variance=1.2) for n in _nodes},
            edges={e: nn.Linear(10, 10) for e in _edges},
            joins={"n1": jsum},
        )
        trace = GraphTrace(g.spec, ["n0", "n1", "n2", "n3"])
        gx = GraphExecutor(g, trace)
        inits = {"n0": torch.rand(5, 10)}
        hints = {"n1": torch.rand(5, 10)}
        res = gx(inits, hints)

        if training:
            assert torch.all(g.node("n0").activity == inits["n0"])
            assert torch.all(
                g.node("n1").activity
                == g.edge("n0", "n1")(g.node("n0").activity)
                + g.edge("n1", "n1")(hints["n1"])
            )
            assert torch.all(
                g.node("n2").activity == g.edge("n1", "n2")(g.node("n1").activity)
            )
            assert torch.all(
                g.node("n3").activity == g.edge("n2", "n3")(g.node("n2").activity)
            )

        assert torch.all(res["n0"] == inits["n0"])
        assert torch.all(
            res["n1"]
            == g.edge("n0", "n1")(g.node("n0").activity)
            + g.edge("n1", "n1")(hints["n1"])
        )
        assert torch.all(res["n2"] == g.edge("n1", "n2")(g.node("n1").activity))
        assert torch.all(res["n3"] == g.edge("n2", "n3")(g.node("n2").activity))

    @pytest.mark.parametrize(
        "exclude_initial",
        (True, False),
        ids=("exclude_initial=True", "exclude_initial=False"),
    )
    def test_named_estep_params(self, exclude_initial):
        _nodes = ["n0", "n1", "n2", "n3"]
        _edges = [("n0", "n1"), ("n1", "n1"), ("n1", "n2"), ("n2", "n3")]

        @eparameters("var")
        class NoisySum(nn.Module):
            var: nn.Parameter

            def __init__(self, var: float) -> None:
                nn.Module.__init__(self)
                self.var = nn.Parameter(torch.tensor(float(var)))

            def forward(self, data: tuple[torch.Tensor, ...]) -> torch.Tensor:
                if not data:
                    res = torch.empty(0)
                else:
                    res = data[0]
                    for d in data[1:]:
                        res = res + d
                return res + (torch.randn_like(res) * self.var)

        @eparameters("var")
        class NoisyLinear(nn.Module):
            var: nn.Parameter
            linear: nn.Linear

            def __init__(self, num_inputs: int, num_outputs: int, var: float) -> None:
                nn.Module.__init__(self)
                self.linear = nn.Linear(num_inputs, num_outputs)
                self.var = nn.Parameter(torch.tensor(float(var)))

        g = Graph(
            nodes={n: FactorizedGaussianNode(10, variance=1.2) for n in _nodes},
            edges={e: NoisyLinear(10, 10, random.uniform(0.5, 1.5)) for e in _edges},
            joins={n: NoisySum(random.uniform(0.5, 1.5)) for n in _nodes},
        )
        trace = GraphTrace(g.spec, ["n0", "n1", "n2", "n3"])
        gx = GraphExecutor(g, trace)

        eparams = {}

        if not exclude_initial:
            eparams |= {
                n: p
                for n, p in get_named_estep_params(
                    g.node("n0"), prefix="graph.nodes.n0"
                )
            }
        eparams |= {
            n: p
            for n, p in get_named_estep_params(
                g.nodes, prefix="graph.nodes", exclude=[g.node("n0")]
            )
        }

        eparams |= {
            n: p for n, p in get_named_estep_params(g.edges, prefix="graph.edges")
        }

        if not exclude_initial:
            eparams |= {
                n: p
                for n, p in get_named_estep_params(
                    g.join("n0"), prefix="graph.joins.n0"
                )
            }
        eparams |= {
            n: p
            for n, p in get_named_estep_params(
                g.joins, prefix="graph.joins", exclude=[g.join("n0")]
            )
        }

        gx_eparams = {n: p for n, p in gx.named_estep_params(exclude_initial)}

        assert eparams.keys() == gx_eparams.keys()
        for k in eparams:
            assert eparams[k] is gx_eparams[k]

    @pytest.mark.parametrize(
        "exclude_initial",
        (True, False),
        ids=("exclude_initial=True", "exclude_initial=False"),
    )
    def test_named_mstep_params(self, exclude_initial):
        _nodes = ["n0", "n1", "n2", "n3"]
        _edges = [("n0", "n1"), ("n1", "n1"), ("n1", "n2"), ("n2", "n3")]

        @mparameters("var")
        class NoisySum(nn.Module):
            var: nn.Parameter

            def __init__(self, var: float) -> None:
                nn.Module.__init__(self)
                self.var = nn.Parameter(torch.tensor(float(var)))

            def forward(self, data: tuple[torch.Tensor, ...]) -> torch.Tensor:
                if not data:
                    res = torch.empty(0)
                else:
                    res = data[0]
                    for d in data[1:]:
                        res = res + d
                return res + (torch.randn_like(res) * self.var)

        @mparameters("var")
        class NoisyLinear(nn.Module):
            var: nn.Parameter
            linear: nn.Linear

            def __init__(self, num_inputs: int, num_outputs: int, var: float) -> None:
                nn.Module.__init__(self)
                self.linear = nn.Linear(num_inputs, num_outputs)
                self.var = nn.Parameter(torch.tensor(float(var)))

        g = Graph(
            nodes={n: FactorizedGaussianNode(10, variance=1.2) for n in _nodes},
            edges={e: NoisyLinear(10, 10, random.uniform(0.5, 1.5)) for e in _edges},
            joins={n: NoisySum(random.uniform(0.5, 1.5)) for n in _nodes},
        )
        trace = GraphTrace(g.spec, ["n0", "n1", "n2", "n3"])
        gx = GraphExecutor(g, trace)

        mparams = {}

        if not exclude_initial:
            mparams |= {
                n: p
                for n, p in get_named_mstep_params(
                    g.node("n0"), prefix="graph.nodes.n0"
                )
            }
        mparams |= {
            n: p
            for n, p in get_named_mstep_params(
                g.nodes, prefix="graph.nodes", exclude=[g.node("n0")]
            )
        }

        mparams |= {
            n: p for n, p in get_named_mstep_params(g.edges, prefix="graph.edges")
        }

        if not exclude_initial:
            mparams |= {
                n: p
                for n, p in get_named_mstep_params(
                    g.join("n0"), prefix="graph.joins.n0"
                )
            }
        mparams |= {
            n: p
            for n, p in get_named_mstep_params(
                g.joins, prefix="graph.joins", exclude=[g.join("n0")]
            )
        }

        gx_mparams = {n: p for n, p in gx.named_mstep_params(exclude_initial)}

        assert mparams.keys() == gx_mparams.keys()
        for k in mparams:
            assert mparams[k] is gx_mparams[k]
