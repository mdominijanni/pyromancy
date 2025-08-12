from collections.abc import Hashable, Sequence
import networkx as nx
import torch


def digraph_from_adjmat[T: Hashable](
    adj: torch.Tensor, nodes: Sequence[T] | None = None
) -> nx.DiGraph:
    assert adj.dtype == torch.bool
    assert adj.ndim == 2
    assert adj.size(0) == adj.size(1)

    if nodes is not None:
        assert len(nodes) == adj.size(0)
        _nn = nodes
    else:
        _nn = [*range(adj.size(0))]

    return nx.DiGraph(
        (_nn[r.item()], _nn[c.item()])
        for r, c in zip(*torch.nonzero(adj, as_tuple=True))
    )


def random_dag[T: Hashable](
    nodes: Sequence[T],
    prob: float = 0.5,
    nretries: int = 100,
    generator: torch.Generator | None = None,
) -> nx.DiGraph:
    assert 0 <= prob <= 1
    for _ in range(nretries):
        adj = torch.randn(len(nodes), len(nodes), generator=generator)
        graph = digraph_from_adjmat(adj.lt(prob).triu(1), nodes)
        if nx.number_weakly_connected_components(graph) == 1:
            return graph
    raise RuntimeError(f"failed to create a DAG in {nretries} tries")
