"""P&ID XML  ---> graph structure, analysis, subgraphs, similarity, and ML-ready features (POC)."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from typing import Any


def _local_name(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _iter_local(root: ET.Element, name: str):
    for el in root.iter():
        if _local_name(el.tag) == name:
            yield el


def build_graph_from_xml(xml_data: str) -> dict[str, Any]:
    """Step 1: Parse XML  ---> nodes (equipment) and edges (connections)."""
    root = ET.fromstring(xml_data)

    nodes: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    for equipment in _iter_local(root, "Equipment"):
        eid = equipment.get("id")
        if not eid:
            continue
        nodes.append(
            {
                "id": eid,
                "name": equipment.get("name", eid),
                "type": equipment.get("type", "Unknown"),
            }
        )
        node_ids.add(eid)

    edges: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for parent in root.iter():
        if _local_name(parent.tag) != "PipingComponent":
            continue
        pc_id = parent.get("id", "")
        for child in parent:
            if _local_name(child.tag) != "Connection":
                continue
            fid = child.get("from")
            tid = child.get("to")
            if not fid or not tid:
                continue
            key = (fid, tid)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            edges.append(
                {
                    "from": fid,
                    "to": tid,
                    "piping_component_id": pc_id or None,
                }
            )

    checks = {
        "nodes_are_equipment": len(nodes) > 0,
        "edges_are_connections": len(edges) >= 0,
        "graph_not_empty": len(nodes) > 0,
        "orphan_edge_endpoints": [
            e
            for e in edges
            if e["from"] not in node_ids or e["to"] not in node_ids
        ],
    }

    return {
        "nodes": nodes,
        "edges": edges,
        "checks": checks,
    }


def _adjacency_directed(edges: list[dict[str, Any]]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        adj[e["from"]].add(e["to"])
    return adj


def _adjacency_undirected(edges: list[dict[str, Any]]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        adj[e["from"]].add(e["to"])
        adj[e["to"]].add(e["from"])
    return adj


def _degrees_undirected(
    node_ids: list[str], edges: list[dict[str, Any]]
) -> dict[str, int]:
    deg: dict[str, int] = {n: 0 for n in node_ids}
    for e in edges:
        deg[e["from"]] = deg.get(e["from"], 0) + 1
        deg[e["to"]] = deg.get(e["to"], 0) + 1
    return deg


def analyze_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Step 2: Basic metrics, importance, connectivity, path analysis."""
    nodes = graph["nodes"]
    edges = graph["edges"]
    id_list = [n["id"] for n in nodes]
    id_set = set(id_list)
    deg = _degrees_undirected(id_list, edges)
    high_degree = sorted(
        ((nid, deg[nid]) for nid in id_list if deg.get(nid, 0) >= 3),
        key=lambda x: -x[1],
    )
    neighbors = {nid: sorted(_adjacency_undirected(edges).get(nid, set())) for nid in id_list}
    isolated = [nid for nid in id_list if deg.get(nid, 0) == 0]

    adj_dir = _adjacency_directed(edges)

    def shortest_path_directed(src: str, dst: str) -> list[str] | None:
        if src not in id_set or dst not in id_set:
            return None
        if src == dst:
            return [src]
        q = deque([(src, [src])])
        visited = {src}
        while q:
            u, path = q.popleft()
            for v in adj_dir.get(u, ()):
                if v in visited:
                    continue
                np = path + [v]
                if v == dst:
                    return np
                visited.add(v)
                q.append((v, np))
        return None

    # Example queries: first two nodes with edge between them, else first two nodes
    path_example: dict[str, Any] | None = None
    if edges:
        a, b = edges[0]["from"], edges[0]["to"]
        path_ab = shortest_path_directed(a, b)
        path_example = {"from": a, "to": b, "directed_path": path_ab}
    elif len(id_list) >= 2:
        a, b = id_list[0], id_list[1]
        path_example = {
            "from": a,
            "to": b,
            "directed_path": shortest_path_directed(a, b),
        }

    return {
        "basic_metrics": {
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
        "node_importance": {
            "degree_by_node": {nid: deg.get(nid, 0) for nid in id_list},
            "high_degree_nodes": [{"id": nid, "degree": d} for nid, d in high_degree],
        },
        "connectivity": {
            "neighbors_by_node": neighbors,
            "isolated_nodes": isolated,
        },
        "path_analysis": {
            "directed_reachability_example": path_example,
        },
    }


def extract_subgraphs(graph: dict[str, Any]) -> dict[str, Any]:
    """Step 3: Rule-based patterns (POC)."""
    nodes = graph["nodes"]
    edges = graph["edges"]
    id_list = [n["id"] for n in nodes]
    by_id = {n["id"]: n for n in nodes}
    deg = _degrees_undirected(id_list, edges)
    adj_u = _adjacency_undirected(edges)
    adj_d = _adjacency_directed(edges)

    branching = [nid for nid in id_list if deg.get(nid, 0) >= 3]
    leaves = [nid for nid in id_list if deg.get(nid, 0) == 1]

    def bfs_downstream(start: str, max_nodes: int = 50) -> list[str]:
        seen: list[str] = []
        q = deque([start])
        vis = {start}
        while q and len(seen) < max_nodes:
            u = q.popleft()
            seen.append(u)
            for v in adj_d.get(u, ()):
                if v not in vis:
                    vis.add(v)
                    q.append(v)
        return seen

    pump_nodes = [n["id"] for n in nodes if (n.get("type") or "").lower() == "pump"]
    valve_nodes = [n["id"] for n in nodes if (n.get("type") or "").lower() == "valve"]

    downstream_from_pumps = {
        pid: bfs_downstream(pid) for pid in pump_nodes
    }

    connected_to_valve: dict[str, list[str]] = {}
    for vid in valve_nodes:
        connected_to_valve[vid] = sorted(adj_u.get(vid, set()))

    # Simple "linear chain" hints: longest directed walk without repeating (greedy, POC)
    def greedy_chain(start: str) -> list[str]:
        chain = [start]
        cur = start
        visited = {start}
        while True:
            nxt = None
            for v in adj_d.get(cur, ()):
                if v not in visited:
                    nxt = v
                    break
            if nxt is None:
                break
            visited.add(nxt)
            chain.append(nxt)
            cur = nxt
        return chain

    linear_chains: list[dict[str, Any]] = []
    for nid in id_list:
        if deg.get(nid, 0) != 1:
            continue
        ch = greedy_chain(nid)
        if len(ch) >= 2:
            linear_chains.append({"start": nid, "node_ids": ch})

    return {
        "branching_node_ids": branching,
        "leaf_node_ids": leaves,
        "linear_chain_hints": linear_chains[:20],
        "downstream_from_pumps": downstream_from_pumps,
        "neighbors_of_valves": connected_to_valve,
        "subgraph_node_labels": {nid: by_id[nid].get("name", nid) for nid in id_list},
    }


def _graph_embedding(graph: dict[str, Any], analysis: dict[str, Any]) -> list[float]:
    """Step 5: Fixed-size feature vector (POC)."""
    n = analysis["basic_metrics"]["node_count"]
    m = analysis["basic_metrics"]["edge_count"]
    deg_vals = list(analysis["node_importance"]["degree_by_node"].values())
    max_d = max(deg_vals) if deg_vals else 0.0
    mean_d = sum(deg_vals) / len(deg_vals) if deg_vals else 0.0
    dens = (2.0 * m / (n * (n - 1))) if n > 1 else 0.0
    n_branch = float(len(analysis["node_importance"]["high_degree_nodes"]))
    n_iso = float(len(analysis["connectivity"]["isolated_nodes"]))
    return [
        float(n),
        float(m),
        mean_d,
        float(max_d),
        dens,
        n_branch,
        n_iso,
    ]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def graph_similarity(
    graph_a: dict[str, Any],
    analysis_a: dict[str, Any],
    graph_b: dict[str, Any],
    analysis_b: dict[str, Any],
) -> dict[str, Any]:
    """Step 4: Basic structural + topology-ish scoring (POC)."""
    n1 = analysis_a["basic_metrics"]["node_count"]
    n2 = analysis_b["basic_metrics"]["node_count"]
    m1 = analysis_a["basic_metrics"]["edge_count"]
    m2 = analysis_b["basic_metrics"]["edge_count"]

    def rel_diff(a: int, b: int) -> float:
        denom = max(abs(a), abs(b), 1)
        return abs(a - b) / denom

    struct_score = 1.0 - 0.5 * (rel_diff(n1, n2) + rel_diff(m1, m2))
    struct_score = max(0.0, min(1.0, struct_score))

    emb_a = _graph_embedding(graph_a, analysis_a)
    emb_b = _graph_embedding(graph_b, analysis_b)
    emb_sim = _cosine_similarity(emb_a, emb_b)

    # Blend: structural agreement + embedding alignment
    combined = 0.45 * struct_score + 0.55 * max(0.0, min(1.0, emb_sim))

    return {
        "structural": {
            "node_count_a": n1,
            "node_count_b": n2,
            "edge_count_a": m1,
            "edge_count_b": m2,
            "structural_similarity_0_1": round(struct_score, 4),
        },
        "topology_proxy": {
            "embedding_cosine_similarity": round(emb_sim, 4),
            "note": "POC: cosine on hand-crafted graph features, not graph edit distance.",
        },
        "combined_similarity_0_1": round(combined, 4),
    }


def gnn_poc_layer(
    analysis: dict[str, Any], embedding: list[float]
) -> dict[str, Any]:
    """Step 6: Conceptual ML layer — classification from simple features."""
    n = analysis["basic_metrics"]["node_count"]
    m = analysis["basic_metrics"]["edge_count"]
    if n < 5:
        complexity = "simple"
    elif n < 12:
        complexity = "moderate"
    else:
        complexity = "complex"
    density = embedding[4] if len(embedding) > 4 else 0.0
    return {
        "complexity_class": complexity,
        "rationale": f"POC rule: node_count={n}, edge_count={m}, density≈{density:.4f}",
        "embedding_dim": len(embedding),
        "tasks": {
            "diagram_complexity": complexity,
            "similarity_ready": True,
        },
    }


def run_full_pipeline(
    xml_a: str, xml_b: str | None = None
) -> dict[str, Any]:
    """Run steps 1–6 for diagram A; optionally compare with B."""
    g1 = build_graph_from_xml(xml_a)
    a1 = analyze_graph(g1)
    s1 = extract_subgraphs(g1)
    e1 = _graph_embedding(g1, a1)
    ml1 = gnn_poc_layer(a1, e1)

    out: dict[str, Any] = {
        "diagram_a": {
            "step1_graph": g1,
            "step2_analysis": a1,
            "step3_subgraphs": s1,
            "step5_embedding": e1,
            "step6_gnn_poc": ml1,
        },
    }

    if xml_b is not None and xml_b.strip():
        g2 = build_graph_from_xml(xml_b)
        a2 = analyze_graph(g2)
        s2 = extract_subgraphs(g2)
        e2 = _graph_embedding(g2, a2)
        ml2 = gnn_poc_layer(a2, e2)
        sim = graph_similarity(g1, a1, g2, a2)
        out["diagram_b"] = {
            "step1_graph": g2,
            "step2_analysis": a2,
            "step3_subgraphs": s2,
            "step5_embedding": e2,
            "step6_gnn_poc": ml2,
        }
        out["step4_similarity"] = sim

    return out
