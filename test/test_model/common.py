import networkx as nx
import torch


def digraph_from_adjmat(adj: torch.Tensor) -> nx.DiGraph:
    assert adj.dtype == torch.bool
    assert adj.ndim == 2
    assert adj.size(0) == adj.size(1)

    return nx.DiGraph(
        (r.item(), c.item()) for r, c in zip(*torch.nonzero(adj, as_tuple=True))
    )


def random_dag(
    nnodes: int,
    prob: float = 0.5,
    nretries: int = 100,
    generator: torch.Generator | None = None,
) -> nx.DiGraph:
    assert 0 <= prob <= 1
    for _ in range(nretries):
        adj = torch.randn(nnodes, nnodes, generator=generator)
        graph = digraph_from_adjmat(adj.lt(prob).triu(1))
        if nx.number_weakly_connected_components(graph) == 1:
            return graph
    raise RuntimeError(f"failed to create a DAG in {nretries} tries")
