import networkx as nx
import numpy as np
import pytest

from src.graph import pagerank


def test_pagerank_matches_networkx_on_a_small_directed_graph():
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 2), (4, 2)]  # node 4 has no in-links
    g = nx.DiGraph(edges)
    expected = nx.pagerank(g, alpha=0.85, tol=1e-12)
    src = np.array([a for a, _ in edges])
    dst = np.array([b for _, b in edges])
    pr = pagerank(src, dst, n=5, alpha=0.85)
    assert pr.sum() == pytest.approx(1.0)
    assert pr.tolist() == pytest.approx([expected[i] for i in range(5)], abs=1e-8)


def test_pagerank_handles_dangling_nodes_like_networkx():
    edges = [(0, 1), (1, 2)]  # node 2 has no out-links
    expected = nx.pagerank(nx.DiGraph(edges), alpha=0.85, tol=1e-12)
    pr = pagerank(np.array([0, 1]), np.array([1, 2]), n=3)
    assert pr.tolist() == pytest.approx([expected[i] for i in range(3)], abs=1e-8)
