import json
import numpy as np
import networkx as nx


def _sanitize_attrs(d):
    for k, v in list(d.items()):
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, (list, tuple, dict)):
            d[k] = json.dumps(v)


def export_graphml(G: nx.DiGraph, path: str):
    G2 = G.copy()

    for n, d in G2.nodes(data=True):
        d["label"] = str(n)
        _sanitize_attrs(d)

    for u, v, d in G2.edges(data=True):
        d["label"] = f"{u}->{v}"
        _sanitize_attrs(d)

    nx.write_graphml(G2, path)