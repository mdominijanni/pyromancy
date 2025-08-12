import einops as ein
import networkx as nx
import pytest
import random
import torch
import torch.nn as nn

from pyromancy.nodes import PredictiveNode, FloatNode, StandardGaussianNode
from pyromancy.model import Graph, GraphSpec

from .common import random_dag


class TestGraphSpec:

    def test_init_badgraph_empty(self):
        graph = nx.DiGraph()
        nodes = []
        edges = []

        with pytest.raises(RuntimeError) as excinfo:
            _ = GraphSpec(graph, nodes, edges)
        assert "`graph` must have exactly one weakly connected component" in str(
            excinfo.value
        )

    def test_init_badgraph_disjoint(self):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C", "D"]
        edges = [("A", "B"), ("C", "D")]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        with pytest.raises(RuntimeError) as excinfo:
            _ = GraphSpec(graph, nodes, edges)
        assert "`graph` must have exactly one weakly connected component" in str(
            excinfo.value
        )

    def test_init_badorder_node_nonunique(self):
        graph = nx.DiGraph()
        nodes = ["A", "B", "A"]
        edges = [("A", "B")]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        with pytest.raises(RuntimeError) as excinfo:
            _ = GraphSpec(graph, nodes, edges)
        assert "`node_order` cannot contain duplicate entries" in str(excinfo.value)

    def test_init_badorder_node_underspec(self):
        graph = nx.DiGraph()
        nodes = ["A"]
        edges = [("A", "B")]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        with pytest.raises(RuntimeError) as excinfo:
            _ = GraphSpec(graph, nodes, edges)
        assert "`node_order` must contain the exactly the same nodes as `graph`" in str(
            excinfo.value
        )

    def test_init_badorder_node_overspec(self):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C"]
        edges = [("A", "B")]

        graph.add_nodes_from(nodes[:-1])
        graph.add_edges_from(edges)

        with pytest.raises(RuntimeError) as excinfo:
            _ = GraphSpec(graph, nodes, edges)
        assert "`node_order` must contain the exactly the same nodes as `graph`" in str(
            excinfo.value
        )

    def test_init_badorder_edge_nonunique(self):
        graph = nx.DiGraph()
        nodes = ["A", "B"]
        edges = [("A", "B"), ("A", "B")]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        with pytest.raises(RuntimeError) as excinfo:
            _ = GraphSpec(graph, nodes, edges)
        assert "`edge_order` cannot contain duplicate entries" in str(excinfo.value)

    def test_init_badorder_edge_underspec(self):
        graph = nx.DiGraph()
        nodes = ["A", "B"]
        edges = [("A", "B"), ("B", "A")]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges[:-1])

        with pytest.raises(RuntimeError) as excinfo:
            _ = GraphSpec(graph, nodes, edges)
        assert "`edge_order` must contain the exactly the same edges as `graph`" in str(
            excinfo.value
        )

    def test_init_badorder_edge_overspec(self):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C"]
        edges = [("A", "B"), ("B", "C")]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges + [("C", "A")])

        with pytest.raises(RuntimeError) as excinfo:
            _ = GraphSpec(graph, nodes, edges)
        assert "`edge_order` must contain the exactly the same edges as `graph`" in str(
            excinfo.value
        )

    def test_eq_diff_class(self):
        graph = nx.DiGraph()
        nodes = ["A", "B"]
        edges = [("A", "B")]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        spec = GraphSpec(graph, nodes, edges)

        assert spec != graph

    def test_eq_diff_graph(self):
        graph = nx.DiGraph()
        nodes = ["A", "B"]
        edges = [("A", "B")]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        other_edges = [("B", "A")]

        otherG = nx.DiGraph()
        otherG.add_nodes_from(nodes)
        otherG.add_edges_from(other_edges)

        spec = GraphSpec(graph, nodes, edges)
        other = GraphSpec(otherG, nodes, other_edges)

        assert spec != other

    def test_eq_diff_nodeorder(self):
        graph = nx.DiGraph()
        nodes = ["A", "B"]
        edges = [("A", "B")]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        spec = GraphSpec(graph, nodes, edges)
        other = GraphSpec(graph, [*reversed(nodes)], edges)

        assert spec != other

    def test_eq_diff_edgeorder(self):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C"]
        edges = [("A", "B"), ("B", "C")]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        spec = GraphSpec(graph, nodes, edges)
        other = GraphSpec(graph, nodes, [*reversed(edges)])

        assert spec != other

    def test_eq_equiv(self):
        graph = nx.DiGraph()
        nodes = ["A", "B"]
        edges = [("A", "B")]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        spec = GraphSpec(graph, nodes, edges)
        other = GraphSpec(graph.copy(), tuple(nodes), tuple(edges))

        assert spec == other

    @pytest.mark.parametrize(
        "reverse_node_order",
        (True, False),
        ids=("reverse_node_order=True", "reverse_node_order=False"),
    )
    @pytest.mark.parametrize(
        "reverse_edge_order",
        (True, False),
        ids=("reverse_edge_order=True", "reverse_edge_order=False"),
    )
    def test_reverse(self, reverse_node_order, reverse_edge_order):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C"]
        edges = [("A", "B"), ("B", "C"), ("C", "A")]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        spec = GraphSpec(graph, nodes, edges)
        rev = spec.reverse(reverse_node_order, reverse_edge_order)

        assert rev._graph.nodes == spec._graph.nodes
        assert rev._graph.edges == spec._graph.reverse().edges

        if reverse_node_order:
            assert all(
                rev._node_order[n] == (len(nodes) - 1 - spec._node_order[n])
                for n in nodes
            )
        else:
            assert all(rev._node_order[n] == spec._node_order[n] for n in nodes)

        if reverse_edge_order:
            assert all(
                rev._edge_order[(e[1], e[0])] == (len(edges) - 1 - spec._edge_order[e])
                for e in edges
            )
        else:
            assert all(
                rev._edge_order[(e[1], e[0])] == spec._edge_order[e] for e in edges
            )

    def test_nodes(self):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C"]
        edges = [("A", "B"), ("B", "C"), ("C", "A")]

        graph.add_nodes_from(reversed(nodes))
        graph.add_edges_from(reversed(edges))

        spec = GraphSpec(graph, nodes, edges)

        assert all(node == sol for node, sol in zip(spec.nodes(), nodes))

    def test_edges(self):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C", "D", "E"]
        edges = [("A", "C"), ("A", "D"), ("A", "E"), ("B", "C"), ("B", "D"), ("B", "E")]

        graph.add_nodes_from(reversed(nodes))
        graph.add_edges_from(reversed(edges))

        spec = GraphSpec(graph, nodes, edges)

        assert all(edge == sol for edge, sol in zip(spec.edges(), edges))

    def test_successors(self):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C", "D", "E"]
        edges = [("A", "C"), ("A", "D"), ("A", "E"), ("B", "C"), ("B", "D"), ("B", "E")]

        graph.add_nodes_from(reversed(nodes))
        graph.add_edges_from(reversed(edges))

        spec = GraphSpec(graph, [*reversed(nodes)], edges)

        assert tuple(spec.successors("A")) == ("C", "D", "E")
        assert tuple(spec.successors("B")) == ("C", "D", "E")

    def test_predecessors(self):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C", "D", "E"]
        edges = [("A", "C"), ("A", "D"), ("A", "E"), ("B", "C"), ("B", "D"), ("B", "E")]

        graph.add_nodes_from(reversed(nodes))
        graph.add_edges_from(reversed(edges))

        spec = GraphSpec(graph, [*reversed(nodes)], edges)

        assert tuple(spec.predecessors("C")) == ("A", "B")
        assert tuple(spec.predecessors("D")) == ("A", "B")
        assert tuple(spec.predecessors("E")) == ("A", "B")

    def test_sort_nodes(self):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C", "D", "E"]
        edges = [("A", "C"), ("A", "D"), ("A", "E"), ("B", "C"), ("B", "D"), ("B", "E")]

        graph.add_nodes_from(reversed(nodes))
        graph.add_edges_from(reversed(edges))

        spec = GraphSpec(graph, nodes, edges)

        assert tuple(spec.sort_nodes(reversed(nodes))) == tuple(nodes)

    def test_sort_edges(self):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C", "D", "E"]
        edges = [("A", "C"), ("A", "D"), ("A", "E"), ("B", "C"), ("B", "D"), ("B", "E")]

        graph.add_nodes_from(reversed(nodes))
        graph.add_edges_from(reversed(edges))

        spec = GraphSpec(graph, nodes, edges)

        assert tuple(spec.sort_edges(reversed(edges))) == tuple(edges)


class TestGraph:
    def test_init_badgraph_nojoin(self):
        with pytest.raises(RuntimeError) as excinfo:
            _ = Graph(
                nodes={
                    "n0": StandardGaussianNode(10),
                    "n1": StandardGaussianNode(10),
                    "n2": StandardGaussianNode(10),
                },
                edges={
                    ("n1", "n0"): nn.Linear(10, 10),
                    ("n2", "n0"): nn.Linear(10, 10),
                },
            )
        assert "`joins` must specify a join for node 'n0' with indegree 2" in str(
            excinfo.value
        )

    def test_init_badgraph_multicomponent(self):
        with pytest.raises(RuntimeError) as excinfo:
            _ = Graph(
                nodes={
                    "n0": StandardGaussianNode(10),
                    "n1": StandardGaussianNode(10),
                    "n2": StandardGaussianNode(10),
                    "n3": StandardGaussianNode(10),
                },
                edges={
                    ("n0", "n1"): nn.Linear(10, 10),
                    ("n2", "n3"): nn.Linear(10, 10),
                },
            )
        assert "`graph` must have exactly one weakly connected component" in str(
            excinfo.value
        )

    def test_init_implicitjoin(self):
        _ = Graph(
            nodes={
                "n0": StandardGaussianNode(10),
                "n1": StandardGaussianNode(10),
                "n2": StandardGaussianNode(10),
            },
            edges={
                ("n0", "n1"): nn.Linear(10, 10),
                ("n1", "n2"): nn.Linear(10, 10),
            },
        )

    def test_spec(self):
        dag = random_dag([f"n{n}" for n in range(7)], 0.2)

        _nodes = [*dag.nodes]
        _edges = [*dag.edges]

        random.shuffle(_nodes)
        random.shuffle(_edges)

        def jsum(data: tuple[torch.Tensor, ...]) -> torch.Tensor:
            if not data:
                res = torch.empty(0)
            else:
                res = data[0]
                for d in data[1:]:
                    res = res + d

            return res

        g = Graph(
            nodes={n: StandardGaussianNode(10) for n in _nodes},
            edges={e: nn.Linear(10, 10) for e in _edges},
            joins={n: jsum for n in _nodes if dag.in_degree(n) > 1},  # type: ignore
        )

        assert g.spec == GraphSpec(dag, _nodes, _edges)

    def test_nodeview(self):
        dag = random_dag([f"n{n}" for n in range(7)], 0.2)

        _nodes = [*dag.nodes]
        _edges = [*dag.edges]

        random.shuffle(_nodes)
        random.shuffle(_edges)

        def jsum(data: tuple[torch.Tensor, ...]) -> torch.Tensor:
            if not data:
                res = torch.empty(0)
            else:
                res = data[0]
                for d in data[1:]:
                    res = res + d

            return res

        g = Graph(
            nodes={n: StandardGaussianNode(10) for n in _nodes},
            edges={e: nn.Linear(10, 10) for e in _edges},
            joins={n: jsum for n in _nodes if dag.in_degree(n) > 1},  # type: ignore
        )

        spec = GraphSpec(dag, _nodes, _edges)

        for n in _nodes:
            nv = g.nodeview(n)
            assert nv._node is g.node(n)
            assert nv._join is g.join(n)
            for (nv_pred, nv_edge), pred in zip(nv._predecessors, spec.predecessors(n)):
                assert nv_pred is g.node(pred)
                assert nv_edge is g.edge(pred, n)

    def test_reset(self):
        dag = random_dag([f"n{n}" for n in range(7)], 0.2)

        _nodes = [*dag.nodes]
        _edges = [*dag.edges]

        random.shuffle(_nodes)
        random.shuffle(_edges)

        def jsum(data: tuple[torch.Tensor, ...]) -> torch.Tensor:
            if not data:
                res = torch.empty(0)
            else:
                res = data[0]
                for d in data[1:]:
                    res = res + d

            return res

        g = Graph(
            nodes={n: StandardGaussianNode(10) for n in _nodes},
            edges={e: nn.Linear(10, 10) for e in _edges},
            joins={n: jsum for n in _nodes if dag.in_degree(n) > 1},  # type: ignore
        )

        for node in g.nodes.values():
            node.init(torch.rand(5, 10))

        assert all(tuple(node.activity.shape) == (5, 10) for node in g.nodes.values())

        g.reset()

        assert all(tuple(node.activity.shape) == (0,) for node in g.nodes.values())

    def test_energy(self):
        dag = random_dag([f"n{n}" for n in range(7)], 0.2)

        _nodes = [*dag.nodes]
        _edges = [*dag.edges]

        random.shuffle(_nodes)
        random.shuffle(_edges)

        def jsum(data: tuple[torch.Tensor, ...]) -> torch.Tensor:
            if not data:
                res = torch.empty(0)
            else:
                res = data[0]
                for d in data[1:]:
                    res = res + d

            return res

        g = Graph(
            nodes={n: StandardGaussianNode(10) for n in _nodes[:4]}
            | {n: FloatNode(10) for n in _nodes[4:]},
            edges={e: nn.Linear(10, 10) for e in _edges},
            joins={n: jsum for n in _nodes if dag.in_degree(n) > 1},  # type: ignore
        )

        spec = GraphSpec(dag, _nodes, _edges)

        for node in g.nodes.values():
            node.init(torch.rand(5, 10))

        energy = []
        for tgt in spec.nodes():
            node = g.node(tgt)
            predecessors = spec.predecessors(tgt)

            if not isinstance(node, PredictiveNode) or not predecessors:
                continue

            pred = g.join(tgt)(
                tuple(g.edge(src, tgt)(g.node(src).activity) for src in predecessors)
            )
            energy.append(node.energy(pred))

        energy = ein.reduce(energy, "n ... -> ...", "sum")

        assert torch.allclose(energy, g.energy())


class TestGraphNodeView:

    def test_node(self):
        dag = random_dag([f"n{n}" for n in range(7)], 0.2)

        _nodes = [*dag.nodes]
        _edges = [*dag.edges]

        random.shuffle(_nodes)
        random.shuffle(_edges)

        def jsum(data: tuple[torch.Tensor, ...]) -> torch.Tensor:
            if not data:
                res = torch.empty(0)
            else:
                res = data[0]
                for d in data[1:]:
                    res = res + d

            return res

        g = Graph(
            nodes={n: StandardGaussianNode(10) for n in _nodes[:4]}
            | {n: FloatNode(10) for n in _nodes[4:]},
            edges={e: nn.Linear(10, 10) for e in _edges},
            joins={n: jsum for n in _nodes if dag.in_degree(n) > 1},  # type: ignore
        )

        assert all(g.nodeview(n).node is g.node(n) for n in _nodes)

    def test_prediction(self):
        dag = random_dag([f"n{n}" for n in range(7)], 0.2)

        _nodes = [*dag.nodes]
        _edges = [*dag.edges]

        random.shuffle(_nodes)
        random.shuffle(_edges)

        def jsum(data: tuple[torch.Tensor, ...]) -> torch.Tensor:
            if not data:
                res = torch.empty(0)
            else:
                res = data[0]
                for d in data[1:]:
                    res = res + d

            return res

        g = Graph(
            nodes={n: StandardGaussianNode(10) for n in _nodes[:4]}
            | {n: FloatNode(10) for n in _nodes[4:]},
            edges={e: nn.Linear(10, 10) for e in _edges},
            joins={n: jsum for n in _nodes if dag.in_degree(n) > 1},  # type: ignore
        )

        spec = GraphSpec(dag, _nodes, _edges)

        for node in g.nodes.values():
            node.init(torch.rand(5, 10))

        for n in _nodes:
            nv = g.nodeview(n)

            if dag.in_degree(n) <= 0:  # type: ignore
                with pytest.raises(RuntimeError) as excinfo:
                    _ = nv.prediction
                assert "cannot call `prediction` on a node without parents" in str(
                    excinfo.value
                )
                continue

            pred = g.join(n)(
                tuple(
                    g.edge(src, n)(g.node(src).activity) for src in spec.predecessors(n)
                )
            )

            assert torch.allclose(pred, nv.prediction)

    def test_error(self):
        dag = random_dag([f"n{n}" for n in range(7)], 0.2)

        _nodes = [*dag.nodes]
        _edges = [*dag.edges]

        random.shuffle(_nodes)
        random.shuffle(_edges)

        def jsum(data: tuple[torch.Tensor, ...]) -> torch.Tensor:
            if not data:
                res = torch.empty(0)
            else:
                res = data[0]
                for d in data[1:]:
                    res = res + d

            return res

        g = Graph(
            nodes={n: StandardGaussianNode(10) for n in _nodes[:4]}
            | {n: FloatNode(10) for n in _nodes[4:]},
            edges={e: nn.Linear(10, 10) for e in _edges},
            joins={n: jsum for n in _nodes if dag.in_degree(n) > 1},  # type: ignore
        )

        spec = GraphSpec(dag, _nodes, _edges)

        for node in g.nodes.values():
            node.init(torch.rand(5, 10))

        for n in _nodes:
            node = g.node(n)
            nv = g.nodeview(n)

            if dag.in_degree(n) <= 0:  # type: ignore
                with pytest.raises(RuntimeError) as excinfo:
                    _ = nv.error
                assert "cannot call `prediction` on a node without parents" in str(
                    excinfo.value
                )
                continue

            pred = g.join(n)(
                tuple(
                    g.edge(src, n)(g.node(src).activity) for src in spec.predecessors(n)
                )
            )

            assert torch.allclose(node.error(pred), nv.error)

    def test_energy(self):
        dag = random_dag([f"n{n}" for n in range(7)], 0.2)

        _nodes = [*dag.nodes]
        _edges = [*dag.edges]

        random.shuffle(_nodes)
        random.shuffle(_edges)

        def jsum(data: tuple[torch.Tensor, ...]) -> torch.Tensor:
            if not data:
                res = torch.empty(0)
            else:
                res = data[0]
                for d in data[1:]:
                    res = res + d

            return res

        g = Graph(
            nodes={n: StandardGaussianNode(10) for n in _nodes[:4]}
            | {n: FloatNode(10) for n in _nodes[4:]},
            edges={e: nn.Linear(10, 10) for e in _edges},
            joins={n: jsum for n in _nodes if dag.in_degree(n) > 1},  # type: ignore
        )

        spec = GraphSpec(dag, _nodes, _edges)

        for node in g.nodes.values():
            node.init(torch.rand(5, 10))

        for n in _nodes:
            node = g.node(n)
            nv = g.nodeview(n)

            if not isinstance(node, PredictiveNode):
                with pytest.raises(TypeError) as excinfo:
                    _ = nv.energy
                assert "only `PredictionNode` nodes support `energy`" in str(
                    excinfo.value
                )
                continue

            if dag.in_degree(n) <= 0:  # type: ignore
                with pytest.raises(RuntimeError) as excinfo:
                    _ = nv.energy
                assert "cannot call `prediction` on a node without parents" in str(
                    excinfo.value
                )
                continue

            pred = g.join(n)(
                tuple(
                    g.edge(src, n)(g.node(src).activity) for src in spec.predecessors(n)
                )
            )

            assert torch.allclose(node.energy(pred), nv.energy)
