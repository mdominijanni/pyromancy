import networkx as nx
import pytest

from pyromancy.model import GraphSpec


class TestGraphSpec:

    def test_init_badgraph_empty(self):
        graph = nx.DiGraph()
        order = []

        with pytest.raises(RuntimeError) as excinfo:
            _ = GraphSpec(graph, order)
        assert "`graph` must have exactly one weakly connected component" in str(
            excinfo.value
        )

    def test_init_badgraph_disjoint(self):
        graph = nx.DiGraph()
        order = ["A", "B", "C", "D"]

        graph.add_nodes_from(order)
        graph.add_edges_from([("A", "B"), ("C", "D")])

        with pytest.raises(RuntimeError) as excinfo:
            _ = GraphSpec(graph, order)
        assert "`graph` must have exactly one weakly connected component" in str(
            excinfo.value
        )

    def test_init_badorder_nonunique(self):
        graph = nx.DiGraph()
        order = ["A", "B", "A"]

        graph.add_nodes_from(order)
        graph.add_edges_from([("A", "B")])

        with pytest.raises(RuntimeError) as excinfo:
            _ = GraphSpec(graph, order)
        assert "order` cannot contain duplicate entries" in str(excinfo.value)

    def test_init_badorder_underspec(self):
        graph = nx.DiGraph()
        order = ["A"]

        graph.add_nodes_from(order)
        graph.add_edges_from([("A", "B")])

        with pytest.raises(RuntimeError) as excinfo:
            _ = GraphSpec(graph, order)
        assert "`order` must contain the exactly the same nodes as `graph`" in str(
            excinfo.value
        )

    def test_init_badorder_overspec(self):
        graph = nx.DiGraph()
        order = ["A", "B", "C"]

        graph.add_nodes_from(order[:-1])
        graph.add_edges_from([("A", "B")])

        with pytest.raises(RuntimeError) as excinfo:
            _ = GraphSpec(graph, order)
        assert "`order` must contain the exactly the same nodes as `graph`" in str(
            excinfo.value
        )

    def test_eq_diffclass(self):
        graph = nx.DiGraph()
        order = ["A", "B"]

        graph.add_nodes_from(order)
        graph.add_edges_from([("A", "B")])

        spec = GraphSpec(graph, order)

        assert spec != graph

    def test_eq_diffgraph(self):
        graph = nx.DiGraph()
        order = ["A", "B"]

        graph.add_nodes_from(order)
        graph.add_edges_from([("A", "B")])

        otherG = nx.DiGraph()
        otherG.add_nodes_from(order)
        otherG.add_edges_from([("B", "A")])

        spec = GraphSpec(graph, order)
        other = GraphSpec(otherG, order)

        assert spec != other

    def test_eq_difforder(self):
        graph = nx.DiGraph()
        order = ["A", "B"]

        graph.add_nodes_from(order)
        graph.add_edges_from([("A", "B")])

        spec = GraphSpec(graph, order)
        other = GraphSpec(graph, [*reversed(order)])

        assert spec != other

    def test_eq_equiv(self):
        graph = nx.DiGraph()
        order = ["A", "B"]

        graph.add_nodes_from(order)
        graph.add_edges_from([("A", "B")])

        spec = GraphSpec(graph, order)
        other = GraphSpec(graph.copy(), tuple(order))

        assert spec == other

    @pytest.mark.parametrize(
        "reverse_order",
        (True, False),
        ids=("reverse_order=True", "reverse_order=False"),
    )
    def test_reverse(self, reverse_order):
        graph = nx.DiGraph()
        order = ["A", "B", "C"]

        graph.add_nodes_from(order)
        graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])

        spec = GraphSpec(graph, order)
        rev = spec.reverse(reverse_order)

        assert rev._graph.nodes == spec._graph.nodes
        assert rev._graph.edges == spec._graph.reverse().edges

        if reverse_order:
            assert all(rev._order[node] == (2 - spec._order[node]) for node in order)
        else:
            assert all(rev._order[node] == spec._order[node] for node in order)

    def test_nodes(self):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C"]
        edges = [("A", "B"), ("B", "C"), ("C", "A")]

        graph.add_nodes_from(reversed(nodes))
        graph.add_edges_from(reversed(edges))

        spec = GraphSpec(graph, nodes)

        assert all(node == sol for node, sol in zip(spec.nodes(), nodes))

    def test_edges(self):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C", "D", "E"]
        edges = [("A", "C"), ("A", "D"), ("A", "E"), ("B", "C"), ("B", "D"), ("B", "E")]

        graph.add_nodes_from(reversed(nodes))
        graph.add_edges_from(reversed(edges))

        spec = GraphSpec(graph, nodes)

        assert all(edge == sol for edge, sol in zip(spec.edges(), edges))

    def test_successors(self):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C", "D", "E"]
        edges = [("A", "C"), ("A", "D"), ("A", "E"), ("B", "C"), ("B", "D"), ("B", "E")]

        graph.add_nodes_from(reversed(nodes))
        graph.add_edges_from(reversed(edges))

        spec = GraphSpec(graph, nodes)

        assert tuple(spec.successors("A")) == ("C", "D", "E")
        assert tuple(spec.successors("B")) == ("C", "D", "E")

    def test_predecessors(self):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C", "D", "E"]
        edges = [("A", "C"), ("A", "D"), ("A", "E"), ("B", "C"), ("B", "D"), ("B", "E")]

        graph.add_nodes_from(reversed(nodes))
        graph.add_edges_from(reversed(edges))

        spec = GraphSpec(graph, nodes)

        assert tuple(spec.predecessors("C")) == ("A", "B")
        assert tuple(spec.predecessors("D")) == ("A", "B")
        assert tuple(spec.predecessors("E")) == ("A", "B")

    def test_sort_nodes(self):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C", "D", "E"]
        edges = [("A", "C"), ("A", "D"), ("A", "E"), ("B", "C"), ("B", "D"), ("B", "E")]

        graph.add_nodes_from(reversed(nodes))
        graph.add_edges_from(reversed(edges))

        spec = GraphSpec(graph, nodes)

        assert tuple(spec.sort_nodes(reversed(nodes))) == tuple(nodes)

    def test_sort_edges(self):
        graph = nx.DiGraph()
        nodes = ["A", "B", "C", "D", "E"]
        edges = [("A", "C"), ("A", "D"), ("A", "E"), ("B", "C"), ("B", "D"), ("B", "E")]

        graph.add_nodes_from(reversed(nodes))
        graph.add_edges_from(reversed(edges))

        spec = GraphSpec(graph, nodes)

        assert tuple(spec.sort_edges(reversed(edges))) == tuple(edges)
