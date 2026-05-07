"""PDF P&ID page -> robust CV pipeline -> DEXPI-like XML -> pydexpi (strict POC)."""

from __future__ import annotations

import base64
import io
import itertools
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore

try:
    import pytesseract
except ImportError:
    pytesseract = None  # type: ignore


def pdf_page_to_png_bytes(pdf_bytes: bytes, page_number: int, dpi: int = 150) -> tuple[bytes, dict[str, Any]]:
    if fitz is None:
        raise RuntimeError("PyMuPDF (pymupdf) is not installed.")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        n = doc.page_count
        if page_number < 1 or page_number > n:
            raise ValueError(f"Page {page_number} is out of range (1-{n}).")
        page = doc.load_page(page_number - 1)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        png = pix.tobytes("png")
        return png, {
            "page_count": n,
            "page_number": page_number,
            "width_px": pix.width,
            "height_px": pix.height,
            "dpi": dpi,
        }
    finally:
        doc.close()


def pdf_page_to_svg_string(pdf_bytes: bytes, page_number: int) -> tuple[str, dict[str, Any]]:
    """Extract one PDF page (1-based) directly as SVG string."""
    if fitz is None:
        raise RuntimeError("PyMuPDF (pymupdf) is not installed.")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        n = doc.page_count
        if page_number < 1 or page_number > n:
            raise ValueError(f"Page {page_number} is out of range (1-{n}).")
        page = doc.load_page(page_number - 1)
        svg = page.get_svg_image(text_as_path=False)
        meta = {
            "page_count": n,
            "page_number": page_number,
            "width_pt": float(page.rect.width),
            "height_pt": float(page.rect.height),
            "method": "pymupdf.get_svg_image",
        }
        return svg, meta
    finally:
        doc.close()


def segment_drawing_area(img_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    ink = (gray < 235).astype(np.uint8) * 255
    nz = cv2.findNonZero(ink)
    if nz is None:
        return img_bgr, {"offset_x": 0, "offset_y": 0, "width": w, "height": h, "title_block_removed": False}
    x, y, bw, bh = cv2.boundingRect(nz)
    pad = max(6, int(0.01 * min(h, w)))
    x0 = max(0, x + pad)
    y0 = max(0, y + pad)
    x1 = min(w, x + bw - pad)
    y1 = min(h, y + bh - pad)
    if x1 <= x0 or y1 <= y0:
        return img_bgr, {"offset_x": 0, "offset_y": 0, "width": w, "height": h, "title_block_removed": False}
    roi = img_bgr[y0:y1, x0:x1]
    rg = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    rid = (rg < 235).astype(np.uint8)
    col_mean = rid.mean(axis=0) if rid.size else np.array([0], dtype=np.float32)
    right_w = max(40, int(0.16 * roi.shape[1]))
    right_density = float(col_mean[-right_w:].mean()) if right_w < len(col_mean) else 0.0
    full_density = float(col_mean.mean()) if len(col_mean) else 0.0
    title_removed = bool(right_density > max(0.025, full_density * 1.35))
    if title_removed:
        roi = roi[:, : roi.shape[1] - right_w]
        x1 = x0 + roi.shape[1]
    return roi, {
        "offset_x": int(x0),
        "offset_y": int(y0),
        "width": int(max(1, x1 - x0)),
        "height": int(max(1, y1 - y0)),
        "title_block_removed": title_removed,
    }


def _circularity(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter * perimeter))


def detect_symbols_bgr(img_bgr: np.ndarray) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (5, 5), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        5,
    )
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    h, w = gray.shape[:2]
    min_a = max(120, int(0.00004 * w * h))
    max_a = int(0.2 * w * h)
    out: list[dict[str, Any]] = []
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    for i in range(1, n_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_a or area > max_a or bw < 12 or bh < 12:
            continue
        ar = bw / max(float(bh), 1.0)
        if ar < 0.2 or ar > 5.0:
            continue
        bbox_area = float(max(1, bw * bh))
        fill = area / bbox_area
        if fill < 0.06:
            continue
        mask = (labels[y : y + bh, x : x + bw] == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        peri = float(cv2.arcLength(c, True))
        circ = _circularity(area, peri)
        # Symbol dictionary (template-like geometric signatures).
        if bh > 2.4 * bw and fill < 0.35:
            st = "Column"
        elif 0.75 <= ar <= 1.25 and circ > 0.62 and fill > 0.35:
            st = "Vessel"
        elif 0.55 <= ar <= 1.8 and 0.42 <= circ <= 0.72 and fill > 0.22:
            st = "Pump"
        elif (bw < 100 and bh < 100 and circ > 0.52) or fill > 0.48:
            st = "Valve"
        elif fill < 0.2:
            st = "Instrument"
        else:
            st = "Equipment"
        out.append(
            {
                "cx": int(x + bw / 2),
                "cy": int(y + bh / 2),
                "bbox": [x, y, bw, bh],
                "area": round(area, 1),
                "circularity": round(circ, 3),
                "fill_ratio": round(fill, 3),
                "type_guess": st,
            }
        )
    out.sort(key=lambda s: -s["area"])
    kept: list[dict[str, Any]] = []
    min_sep = max(12, int(0.012 * min(w, h)))
    for s in out:
        if all((s["cx"] - t["cx"]) ** 2 + (s["cy"] - t["cy"]) ** 2 >= min_sep * min_sep for t in kept):
            kept.append(s)
    return kept


def detect_line_segments_bgr(img_bgr: np.ndarray) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    h, w = gray.shape[:2]
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=max(20, int(0.04 * min(w, h))),
        maxLineGap=int(0.015 * min(w, h)),
    )
    if lines is None:
        return []
    raw: list[dict[str, Any]] = []
    min_len_keep = max(16, int(0.02 * min(w, h)))
    for ln in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in ln]
        dx = x2 - x1
        dy = y2 - y1
        seg_len = float(np.hypot(dx, dy))
        if seg_len < min_len_keep:
            continue
        slope = min(abs(dy) / max(abs(dx), 1), abs(dx) / max(abs(dy), 1))
        if slope > 0.25:
            continue
        raw.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "length": round(seg_len, 1)})
    if not raw:
        return []
    q = max(8, int(0.008 * min(w, h)))
    bins: dict[tuple[int, int], int] = {}
    for s in raw:
        for px, py in ((s["x1"], s["y1"]), (s["x2"], s["y2"])):
            k = (int(round(px / q)), int(round(py / q)))
            bins[k] = bins.get(k, 0) + 1
    return [
        s
        for s in raw
        if bins.get((int(round(s["x1"] / q)), int(round(s["y1"] / q))), 0) >= 2
        or bins.get((int(round(s["x2"] / q)), int(round(s["y2"] / q))), 0) >= 2
        or s["length"] >= 0.12 * min(w, h)
    ]


def _offset_detections(
    symbols: list[dict[str, Any]], segments: list[dict[str, Any]], off_x: int, off_y: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    syms = []
    for s in symbols:
        x, y, bw, bh = s["bbox"]
        ns = dict(s)
        ns["cx"] = int(ns["cx"] + off_x)
        ns["cy"] = int(ns["cy"] + off_y)
        ns["bbox"] = [int(x + off_x), int(y + off_y), int(bw), int(bh)]
        syms.append(ns)
    segs = []
    for g in segments:
        ng = dict(g)
        ng["x1"] = int(ng["x1"] + off_x)
        ng["y1"] = int(ng["y1"] + off_y)
        ng["x2"] = int(ng["x2"] + off_x)
        ng["y2"] = int(ng["y2"] + off_y)
        segs.append(ng)
    return syms, segs


def _clean_ocr_text(s: str) -> str:
    return " ".join((s or "").strip().split())


def extract_symbol_labels_ocr(
    img_bgr: np.ndarray,
    symbols: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    OCR text near each detected symbol and attach `label_text`.
    Returns (updated_symbols, ocr_meta).
    """
    if pytesseract is None:
        return symbols, {"enabled": False, "reason": "pytesseract not installed"}
    if not symbols:
        return symbols, {"enabled": True, "words_detected": 0, "labels_assigned": 0}

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Slight denoise + binary helps OCR for light engineering text.
    prep = cv2.GaussianBlur(gray, (3, 3), 0)
    prep = cv2.adaptiveThreshold(
        prep,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    try:
        data = pytesseract.image_to_data(
            prep,
            output_type=pytesseract.Output.DICT,
            config="--oem 3 --psm 11",
        )
    except Exception as exc:
        return symbols, {"enabled": False, "reason": f"tesseract runtime unavailable: {exc}"}

    words: list[dict[str, Any]] = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = _clean_ocr_text(str(data["text"][i]))
        if not txt:
            continue
        try:
            conf = float(data.get("conf", ["-1"])[i])
        except Exception:
            conf = -1.0
        if conf < 35:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        words.append(
            {
                "text": txt,
                "conf": conf,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": x + w / 2.0,
                "cy": y + h / 2.0,
            }
        )

    updated: list[dict[str, Any]] = []
    labels_assigned = 0
    for s in symbols:
        x, y, bw, bh = [int(v) for v in s["bbox"]]
        cx = float(s["cx"])
        cy = float(s["cy"])
        # Search around symbol including nearby callouts.
        pad_x = max(40, int(2.0 * bw))
        pad_y = max(26, int(1.4 * bh))
        rx0 = x - pad_x
        ry0 = y - pad_y
        rx1 = x + bw + pad_x
        ry1 = y + bh + pad_y
        near = []
        for w in words:
            wcx = float(w["cx"])
            wcy = float(w["cy"])
            if rx0 <= wcx <= rx1 and ry0 <= wcy <= ry1:
                # Keep words not deeply inside filled symbol body; typically labels are adjacent.
                near.append(w)
        near.sort(key=lambda w: _dist(cx, cy, float(w["cx"]), float(w["cy"])))
        label = ""
        if near:
            picked = near[:4]
            label = _clean_ocr_text(" ".join([w["text"] for w in picked]))
        ns = dict(s)
        ns["label_text"] = label
        if label:
            labels_assigned += 1
        updated.append(ns)

    return updated, {
        "enabled": True,
        "words_detected": len(words),
        "labels_assigned": labels_assigned,
    }


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return float(np.hypot(ax - bx, ay - by))


def _quant(x: int, y: int, q: int) -> tuple[int, int]:
    return (int(round(x / q) * q), int(round(y / q) * q))


def _segment_intersection(s1: dict[str, Any], s2: dict[str, Any], tol: int) -> tuple[int, int] | None:
    x1, y1, x2, y2 = s1["x1"], s1["y1"], s1["x2"], s1["y2"]
    x3, y3, x4, y4 = s2["x1"], s2["y1"], s2["x2"], s2["y2"]
    s1_h, s1_v = abs(y1 - y2) <= tol, abs(x1 - x2) <= tol
    s2_h, s2_v = abs(y3 - y4) <= tol, abs(x3 - x4) <= tol
    if s1_h and s2_v:
        ix, iy = int(round((x3 + x4) / 2)), int(round((y1 + y2) / 2))
        if min(x1, x2) - tol <= ix <= max(x1, x2) + tol and min(y3, y4) - tol <= iy <= max(y3, y4) + tol:
            return (ix, iy)
    if s1_v and s2_h:
        ix, iy = int(round((x1 + x2) / 2)), int(round((y3 + y4) / 2))
        if min(x3, x4) - tol <= ix <= max(x3, x4) + tol and min(y1, y2) - tol <= iy <= max(y1, y2) + tol:
            return (ix, iy)
    return None


def _build_line_topology(segments: list[dict[str, Any]], img_shape: tuple[int, int, int]) -> dict[tuple[int, int], set[tuple[int, int]]]:
    h, w = img_shape[:2]
    q = max(6, int(0.006 * min(h, w)))
    nodes: set[tuple[int, int]] = set()
    for s in segments:
        nodes.add(_quant(int(s["x1"]), int(s["y1"]), q))
        nodes.add(_quant(int(s["x2"]), int(s["y2"]), q))
    for a, b in itertools.combinations(segments, 2):
        ip = _segment_intersection(a, b, tol=max(3, q // 2))
        if ip is not None:
            nodes.add(_quant(ip[0], ip[1], q))
    adj: dict[tuple[int, int], set[tuple[int, int]]] = {n: set() for n in nodes}
    for s in segments:
        p1 = _quant(int(s["x1"]), int(s["y1"]), q)
        p2 = _quant(int(s["x2"]), int(s["y2"]), q)
        adj.setdefault(p1, set()).add(p2)
        adj.setdefault(p2, set()).add(p1)
    return adj


def build_graph_from_detections(
    symbols: list[dict[str, Any]], segments: list[dict[str, Any]], img_shape: tuple[int, int, int], max_hook: float | None = None
) -> dict[str, Any]:
    if not symbols:
        return {"nodes": [], "edges": [], "checks": {"note": "No symbols detected."}}
    avg_size = float(np.mean([max(s["bbox"][2], s["bbox"][3]) for s in symbols]))
    hook = max_hook if max_hook is not None else max(28.0, 1.8 * avg_size)
    nodes = [
        {
            "id": f"E{i + 1}",
            "name": (s.get("label_text") or f"{s['type_guess']} E{i + 1}"),
            "type": s["type_guess"],
            "position": {"x": s["cx"], "y": s["cy"]},
            "detection": s,
        }
        for i, s in enumerate(symbols)
    ]
    adj = _build_line_topology(segments, img_shape)
    line_nodes = list(adj.keys())
    if not line_nodes:
        return {"nodes": nodes, "edges": [], "checks": {"hook_distance_px": round(hook, 1), "note": "No line topology."}}

    def near_line_node(px: float, py: float) -> tuple[tuple[int, int] | None, float]:
        best: tuple[tuple[int, int] | None, float] = (None, 1e18)
        for nx, ny in line_nodes:
            d = _dist(px, py, float(nx), float(ny))
            if d < best[1]:
                best = ((nx, ny), d)
        return best

    attach: dict[str, list[tuple[int, int]]] = {}
    port_stats: dict[str, int] = {}
    for n in nodes:
        x = float(n["position"]["x"])
        y = float(n["position"]["y"])
        bw = float(n["detection"]["bbox"][2])
        bh = float(n["detection"]["bbox"][3])
        ports = [(x, y), (x - bw * 0.5, y), (x + bw * 0.5, y), (x, y - bh * 0.5), (x, y + bh * 0.5)]
        linked: list[tuple[int, int]] = []
        for px, py in ports:
            ln, d = near_line_node(px, py)
            if ln is not None and d <= hook:
                linked.append(ln)
        linked = list(dict.fromkeys(linked))
        attach[n["id"]] = linked
        port_stats[n["id"]] = len(linked)

    def connected(a: tuple[int, int], b: tuple[int, int]) -> bool:
        if a == b:
            return True
        q = [a]
        vis = {a}
        while q and len(vis) < 8000:
            u = q.pop(0)
            for v in adj.get(u, ()):
                if v in vis:
                    continue
                if v == b:
                    return True
                vis.add(v)
                q.append(v)
        return False

    ids = [n["id"] for n in nodes]
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if not attach.get(a) or not attach.get(b):
                continue
            linked = any(connected(pa, pb) for pa in attach[a] for pb in attach[b])
            if not linked:
                continue
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            edges.append({"from": a, "to": b, "piping_component_id": f"PC{len(edges)+1}", "via_topology": True})

    max_edges = max(8, 8 * len(nodes))
    if len(edges) > max_edges:
        edges = edges[:max_edges]
    node_id_set = {n["id"] for n in nodes}
    return {
        "nodes": nodes,
        "edges": edges,
        "checks": {
            "nodes_are_equipment": len(nodes) > 0,
            "edges_are_connections": True,
            "graph_not_empty": len(nodes) > 0,
            "orphan_edge_endpoints": [e for e in edges if e["from"] not in node_id_set or e["to"] not in node_id_set],
            "hook_distance_px": round(hook, 1),
            "line_topology_nodes": len(line_nodes),
            "symbol_ports_attached": port_stats,
        },
    }


def evaluate_quality_gates(
    symbols: list[dict[str, Any]], segments: list[dict[str, Any]], graph: dict[str, Any], img_shape: tuple[int, int, int]
) -> dict[str, Any]:
    h, w = img_shape[:2]
    area = float(max(1, h * w))
    sym_density = len(symbols) / (area / 1_000_000.0)
    seg_density = len(segments) / (area / 1_000_000.0)
    gnodes = len(graph.get("nodes", []))
    gedges = len(graph.get("edges", []))
    attach_stats = (graph.get("checks", {}) or {}).get("symbol_ports_attached", {})
    attached_ratio = (
        sum(1 for v in attach_stats.values() if int(v) > 0) / max(1, len(attach_stats)) if attach_stats else 0.0
    )
    checks = {
        "symbols_min": gnodes >= 5,
        "segments_min": len(segments) >= 20,
        "symbols_not_exploded": sym_density < 1200,
        "segments_not_exploded": seg_density < 6000,
        "edges_reasonable": gedges >= 3 and gedges <= max(8, 10 * gnodes),
        "ports_attached_ratio": attached_ratio >= 0.45,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "symbol_count": gnodes,
            "segment_count": len(segments),
            "edge_count": gedges,
            "symbol_density_per_mp": round(sym_density, 2),
            "segment_density_per_mp": round(seg_density, 2),
            "port_attached_ratio": round(attached_ratio, 3),
        },
    }


def build_detection_overlay_png_b64(img_bgr: np.ndarray, symbols: list[dict[str, Any]], segments: list[dict[str, Any]]) -> str:
    vis = img_bgr.copy()
    for seg in segments:
        cv2.line(vis, (int(seg["x1"]), int(seg["y1"])), (int(seg["x2"]), int(seg["y2"])), (35, 82, 214), 1, cv2.LINE_AA)
    for s in symbols:
        x, y, bw, bh = [int(v) for v in s["bbox"]]
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (16, 185, 129), 2)
        cv2.circle(vis, (int(s["cx"]), int(s["cy"])), 4, (16, 185, 129), -1)
    ok, enc = cv2.imencode(".png", vis)
    return base64.b64encode(enc.tobytes()).decode("ascii") if ok else ""


def render_simple_graph_svg(graph: dict[str, Any]) -> str:
    """Render a lightweight graph SVG from extracted node positions + edges."""
    nodes = graph.get("nodes", []) or []
    edges = graph.get("edges", []) or []
    if not nodes:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="240" viewBox="0 0 640 240">'
            '<rect width="640" height="240" fill="white"/>'
            '<text x="20" y="40" font-family="Arial,sans-serif" font-size="16" fill="#111827">'
            "No nodes detected"
            "</text></svg>"
        )
    xs = [float((n.get("position") or {}).get("x", 0)) for n in nodes]
    ys = [float((n.get("position") or {}).get("y", 0)) for n in nodes]
    min_x, max_x = min(xs) - 60, max(xs) + 60
    min_y, max_y = min(ys) - 60, max(ys) + 60
    w = max(300.0, max_x - min_x)
    h = max(220.0, max_y - min_y)

    by_id = {n.get("id"): n for n in nodes}
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(w)}" height="{int(h)}" viewBox="{min_x} {min_y} {w} {h}">',
        f'<rect x="{min_x}" y="{min_y}" width="{w}" height="{h}" fill="white"/>',
    ]
    for e in edges:
        a = by_id.get(e.get("from"))
        b = by_id.get(e.get("to"))
        if not a or not b:
            continue
        ax = float((a.get("position") or {}).get("x", 0))
        ay = float((a.get("position") or {}).get("y", 0))
        bx = float((b.get("position") or {}).get("x", 0))
        by = float((b.get("position") or {}).get("y", 0))
        out.append(
            f'<line x1="{ax}" y1="{ay}" x2="{bx}" y2="{by}" '
            'stroke="#2563eb" stroke-width="2.5" opacity="0.9" />'
        )
    for n in nodes:
        p = n.get("position") or {}
        x = float(p.get("x", 0))
        y = float(p.get("y", 0))
        lbl = n.get("label") or n.get("name") or n.get("id") or "N"
        out.append(f'<circle cx="{x}" cy="{y}" r="12" fill="#f3f4f6" stroke="#111827" stroke-width="1.5" />')
        out.append(
            f'<text x="{x + 16}" y="{y + 4}" font-family="Arial,sans-serif" font-size="11" fill="#111827">{lbl}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def render_symbol_label_overlay_svg(
    image_width: int,
    image_height: int,
    symbols: list[dict[str, Any]],
) -> str:
    """Render symbol bounding boxes and labels in original page coordinates."""
    w = max(1, int(image_width))
    h = max(1, int(image_height))
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect x="0" y="0" width="{w}" height="{h}" fill="white" opacity="0"/>',
    ]
    for i, s in enumerate(symbols, start=1):
        x, y, bw, bh = [int(v) for v in s.get("bbox", [0, 0, 0, 0])]
        label = str(s.get("label_text") or "").strip()
        typ = str(s.get("type_guess") or "Unknown")
        shown = label if label else f"{typ} {i}"
        out.append(
            f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" fill="none" stroke="#dc2626" stroke-width="2"/>'
        )
        tx = x
        ty = max(12, y - 6)
        out.append(
            f'<text x="{tx}" y="{ty}" font-family="Arial,sans-serif" font-size="12" fill="#111827">{shown}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def graph_to_dexpi_like_xml(graph: dict[str, Any], plant_name: str = "From PDF") -> str:
    """Emit minimal DEXPI-like PlantModel XML (matches sample.xml shape for pyDEXPI)."""
    root = ET.Element(
        "PlantModel",
        {"id": "PM1", "name": plant_name},
    )
    pi = ET.SubElement(
        root,
        "PlantInformation",
        {
            "Application": "Dexpi",
            "ApplicationVersion": "1.3.1",
            "Discipline": "PID",
            "Is3D": "no",
            "SchemaVersion": "4.1.1",
            "Units": "px",
            "Date": "2026-05-05",
            "Time": "12:00:00",
            "OriginatingSystem": "pAndIDDiagrams",
            "OriginatingSystemVendor": "pdf_pipeline_poc",
            "OriginatingSystemVersion": "0.1",
        },
    )
    ET.SubElement(pi, "UnitsOfMeasure")

    for n in graph.get("nodes", []):
        pos = n.get("position") or {"x": 0, "y": 0}
        eq = ET.SubElement(
            root,
            "Equipment",
            {"id": n["id"], "name": n.get("name", n["id"]), "type": n.get("type", "Equipment")},
        )
        ET.SubElement(eq, "Position", {"x": str(int(pos["x"])), "y": str(int(pos["y"]))})

    pns = ET.SubElement(root, "PipingNetworkSystem", {"id": "PNS1"})
    for e in graph.get("edges", []):
        pc_id = e.get("piping_component_id") or "PC"
        pc = ET.SubElement(pns, "PipingComponent", {"id": str(pc_id), "type": "Pipe"})
        ET.SubElement(pc, "Connection", {"from": e["from"], "to": e["to"]})

    buf = io.BytesIO()
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue().decode("utf-8")


def load_pydexpi_model(xml_string: str) -> tuple[Any | None, str | None]:
    """Return (model, error)."""
    try:
        from pydexpi.loaders import ProteusSerializer  # type: ignore
    except ModuleNotFoundError:
        return None, "pydexpi not installed."
    loader = ProteusSerializer()
    try:
        if hasattr(loader, "load_from_string"):
            model = loader.load_from_string(xml_string)
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".xml", delete=False, encoding="utf-8"
            ) as f:
                f.write(xml_string)
                tmp = Path(f.name)
            try:
                model = loader.load(tmp.parent, tmp.name)
            finally:
                tmp.unlink(missing_ok=True)
    except Exception as exc:
        return None, str(exc)
    return model, None


def summarize_pydexpi_model(model: Any) -> dict[str, Any]:
    """Lightweight introspection for JSON API."""
    out: dict[str, Any] = {"type": type(model).__name__, "module": type(model).__module__}
    for attr in ("equipment", "piping_network_systems", "plant_information"):
        if hasattr(model, attr):
            try:
                val = getattr(model, attr)
                if val is None:
                    out[attr] = None
                elif hasattr(val, "__len__"):
                    out[attr] = f"len={len(val)}"
                else:
                    out[attr] = repr(val)[:200]
            except Exception as exc:
                out[attr] = f"error: {exc}"
    return out


def run_pdf_page_pipeline(
    pdf_bytes: bytes,
    page_number: int,
    dpi: int = 150,
    strict_quality: bool = True,
) -> dict[str, Any]:
    """Full pipeline: PDF → image → symbols → lines → graph → XML → pydexpi."""
    png_bytes, pdf_meta = pdf_page_to_png_bytes(pdf_bytes, page_number, dpi=dpi)
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode rendered PNG.")

    roi, seg_meta = segment_drawing_area(img)
    symbols_roi = detect_symbols_bgr(roi)
    segments_roi = detect_line_segments_bgr(roi)
    symbols, segments = _offset_detections(
        symbols_roi, segments_roi, int(seg_meta.get("offset_x", 0)), int(seg_meta.get("offset_y", 0))
    )
    symbols, ocr_meta = extract_symbol_labels_ocr(img, symbols)
    debug_overlay_b64 = build_detection_overlay_png_b64(img, symbols, segments)
    graph = build_graph_from_detections(symbols, segments, img.shape)
    quality = evaluate_quality_gates(symbols, segments, graph, img.shape)
    if strict_quality and not quality["passed"]:
        raise ValueError(
            "Strict quality gates failed. "
            + f"Checks={quality['checks']} Metrics={quality['metrics']}"
        )
    xml_str = graph_to_dexpi_like_xml(graph, plant_name=f"PDF page {page_number}")

    model, perr = load_pydexpi_model(xml_str)
    pydexpi_block: dict[str, Any] = {
        "loaded": model is not None,
        "error": perr,
        "summary": summarize_pydexpi_model(model) if model is not None else None,
    }

    preview_b64 = base64.b64encode(png_bytes).decode("ascii")

    # Strip heavy detection copies from graph nodes for smaller JSON (keep positions)
    graph_light = {
        "nodes": [
            {
                "id": n["id"],
                "name": n["name"],
                "type": n["type"],
                "position": n["position"],
            }
            for n in graph["nodes"]
        ],
        "edges": [
            {"from": e["from"], "to": e["to"], "piping_component_id": e.get("piping_component_id")}
            for e in graph["edges"]
        ],
        "checks": graph.get("checks", {}),
    }

    return {
        "pdf_meta": pdf_meta,
        "step_image": {
            "format": "png",
            "preview_base64": preview_b64,
            "note": "Single page rasterized for CV (POC).",
        },
        "step_debug": {
            "overlay_base64": debug_overlay_b64,
            "note": "Debug overlay: red=line segments, green=symbol boxes/centers.",
            "segmentation": seg_meta,
            "quality": quality,
            "ocr": ocr_meta,
        },
        "step_symbols": {
            "count": len(symbols),
            "symbols": symbols,
            "note": "Geometry dictionary classifier (POC).",
        },
        "step_connections": {
            "segment_count": len(segments),
            "segments_sample": segments[:80],
            "note": "Filtered Hough + topology tracing via junctions/ports.",
        },
        "step_graph": graph_light,
        "step_xml": {"dexpi_like": xml_str},
        "step_pydexpi": pydexpi_block,
        "step_simple_graph_svg": {"svg": render_simple_graph_svg(graph_light)},
        "step_symbol_overlay_svg": {
            "svg": render_symbol_label_overlay_svg(
                int(pdf_meta.get("width_px", 1)),
                int(pdf_meta.get("height_px", 1)),
                symbols,
            )
        },
    }
