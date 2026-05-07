"""Minimal Flask POC: upload DEXPI/Proteus XML, render SVG in memory, show in browser."""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
import re
import math
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, url_for

from pid_graph import run_full_pipeline
from pdf_pipeline import (
    _build_line_topology,
    detect_line_segments_bgr,
    pdf_page_to_png_bytes,
    pdf_page_to_svg_string,
    run_pdf_page_pipeline,
    segment_drawing_area,
)
from test import render_fallback_svg

try:
    from pydexpi.loaders import ProteusSerializer  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    ProteusSerializer = None

try:
    from pydexpi.renderers import SvgRenderer  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    SvgRenderer = None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 48 * 1024 * 1024  # 48 MiB (PDF rasterization)


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if not m:
            return None
        return float(m.group(0))
    except Exception:
        return None


def _quant_point(x: float, y: float, grid: float) -> tuple[int, int]:
    g = max(1.0, grid)
    return (int(round(x / g)), int(round(y / g)))


def _estimate_visual_analytics_from_svg(svg_text: str, width_pt: float, height_pt: float) -> dict[str, object]:
    def _extract_path_segments(d_attr: str) -> list[tuple[float, float, float, float]]:
        segs: list[tuple[float, float, float, float]] = []
        tokens = re.findall(r"[A-Za-z]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", d_attr or "")
        if not tokens:
            return segs
        i = 0
        cmd = ""
        cur_x = 0.0
        cur_y = 0.0
        start_x = 0.0
        start_y = 0.0
        while i < len(tokens):
            tk = tokens[i]
            if re.fullmatch(r"[A-Za-z]", tk):
                cmd = tk
                i += 1
                if cmd in ("Z", "z"):
                    segs.append((cur_x, cur_y, start_x, start_y))
                    cur_x, cur_y = start_x, start_y
                continue
            if cmd in ("M", "L"):
                if i + 1 >= len(tokens):
                    break
                nx = float(tokens[i])
                ny = float(tokens[i + 1])
                if cmd == "M":
                    start_x, start_y = nx, ny
                else:
                    segs.append((cur_x, cur_y, nx, ny))
                cur_x, cur_y = nx, ny
                i += 2
            elif cmd in ("m", "l"):
                if i + 1 >= len(tokens):
                    break
                nx = cur_x + float(tokens[i])
                ny = cur_y + float(tokens[i + 1])
                if cmd == "m":
                    start_x, start_y = nx, ny
                else:
                    segs.append((cur_x, cur_y, nx, ny))
                cur_x, cur_y = nx, ny
                i += 2
            elif cmd in ("H",):
                nx = float(tokens[i])
                segs.append((cur_x, cur_y, nx, cur_y))
                cur_x = nx
                i += 1
            elif cmd in ("h",):
                nx = cur_x + float(tokens[i])
                segs.append((cur_x, cur_y, nx, cur_y))
                cur_x = nx
                i += 1
            elif cmd in ("V",):
                ny = float(tokens[i])
                segs.append((cur_x, cur_y, cur_x, ny))
                cur_y = ny
                i += 1
            elif cmd in ("v",):
                ny = cur_y + float(tokens[i])
                segs.append((cur_x, cur_y, cur_x, ny))
                cur_y = ny
                i += 1
            else:
                i += 1
        return segs

    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        line_count_fallback = len(re.findall(r"<\s*line\b", svg_text, flags=re.IGNORECASE))
        path_count_fallback = len(re.findall(r"<\s*path\b", svg_text, flags=re.IGNORECASE))
        text_count_fallback = len(re.findall(r"<\s*text\b", svg_text, flags=re.IGNORECASE))
        return {
            "line_count": line_count_fallback + path_count_fallback,
            "line_element_count": line_count_fallback,
            "path_count": path_count_fallback,
            "text_count": text_count_fallback,
            "junction_count": 0,
            "connected_regions": 0,
            "total_path_length": 0.0,
            "note": "SVG XML parse failed; used regex element counts fallback.",
        }

    line_element_count = 0
    path_count = 0
    text_count = 0
    segment_lengths: list[float] = []
    adj: dict[tuple[int, int], set[tuple[int, int]]] = {}

    diag = max(1.0, math.hypot(max(1.0, width_pt), max(1.0, height_pt)))
    snap_grid = max(1.0, diag / 500.0)
    point_split = re.compile(r"[,\s]+")

    def add_edge(x1: float, y1: float, x2: float, y2: float) -> None:
        p1 = _quant_point(x1, y1, snap_grid)
        p2 = _quant_point(x2, y2, snap_grid)
        if p1 == p2:
            return
        dx = x2 - x1
        dy = y2 - y1
        seg_len = math.hypot(dx, dy)
        if seg_len <= 0:
            return
        segment_lengths.append(seg_len)
        adj.setdefault(p1, set()).add(p2)
        adj.setdefault(p2, set()).add(p1)

    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        if tag == "text":
            text_count += 1

        if tag == "line":
            line_element_count += 1
            x1 = _to_float(elem.attrib.get("x1"))
            y1 = _to_float(elem.attrib.get("y1"))
            x2 = _to_float(elem.attrib.get("x2"))
            y2 = _to_float(elem.attrib.get("y2"))
            if None not in (x1, y1, x2, y2):
                add_edge(float(x1), float(y1), float(x2), float(y2))
        elif tag == "path":
            path_count += 1
            for x1, y1, x2, y2 in _extract_path_segments(str(elem.attrib.get("d") or "")):
                add_edge(x1, y1, x2, y2)
        elif tag in ("polyline", "polygon"):
            raw_points = str(elem.attrib.get("points") or "").strip()
            if not raw_points:
                continue
            vals: list[float] = []
            for tok in point_split.split(raw_points):
                v = _to_float(tok)
                if v is not None:
                    vals.append(v)
            if len(vals) < 4:
                continue
            pts = [(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
            for i in range(len(pts) - 1):
                add_edge(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
            if tag == "polygon" and len(pts) > 2:
                add_edge(pts[-1][0], pts[-1][1], pts[0][0], pts[0][1])
        elif tag == "rect":
            x = _to_float(elem.attrib.get("x")) or 0.0
            y = _to_float(elem.attrib.get("y")) or 0.0
            w = _to_float(elem.attrib.get("width")) or 0.0
            h = _to_float(elem.attrib.get("height")) or 0.0
            if w > 0 and h > 0:
                add_edge(x, y, x + w, y)
                add_edge(x + w, y, x + w, y + h)
                add_edge(x + w, y + h, x, y + h)
                add_edge(x, y + h, x, y)

    line_count = len(segment_lengths)
    degrees = [len(nbrs) for nbrs in adj.values()]
    junction_count = sum(1 for d in degrees if d >= 3)

    visited: set[tuple[int, int]] = set()
    connected_regions = 0
    for n in adj:
        if n in visited:
            continue
        connected_regions += 1
        stack = [n]
        visited.add(n)
        while stack:
            cur = stack.pop()
            for nxt in adj.get(cur, ()):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)

    area = max(1.0, width_pt * height_pt)
    total_path_length = sum(segment_lengths)

    return {
        "line_count": line_count,
        "line_element_count": line_element_count,
        "path_count": path_count,
        "text_count": text_count,
        "junction_count": junction_count,
        "connected_regions": connected_regions,
        "total_path_length": round(total_path_length, 2),
        "line_density_per_1000pt2": round(line_count / (area / 1000.0), 3),
        "text_density_per_1000pt2": round(text_count / (area / 1000.0), 3),
    }


def _estimate_visual_metrics_from_image(pdf_bytes: bytes, page_num: int, dpi: int = 300) -> dict[str, object]:
    """
    Extract level-1 measurable metrics from the rasterized page image.
    """
    png_bytes, _ = pdf_page_to_png_bytes(pdf_bytes, page_num, dpi=dpi)
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {
            "line_count": 0,
            "text_count": 0,
            "junction_count": 0,
            "connected_regions": 0,
            "total_path_length": 0,
            "note": "Image decode failed.",
        }

    roi, _seg_meta = segment_drawing_area(img)
    roi_h, roi_w = roi.shape[:2]
    roi_area_px = max(1, roi_h * roi_w)
    segments = detect_line_segments_bgr(roi)
    line_count = len(segments)
    total_path_length = int(round(sum(float(s.get("length", 0.0)) for s in segments)))
    avg_segment_length = round(total_path_length / max(1, line_count), 2)

    topology = _build_line_topology(segments, roi.shape)
    degrees = [len(nbrs) for nbrs in topology.values()]
    junction_count = sum(1 for d in degrees if d >= 3)

    visited: set[tuple[int, int]] = set()
    connected_regions = 0
    for n in topology:
        if n in visited:
            continue
        connected_regions += 1
        stack = [n]
        visited.add(n)
        while stack:
            cur = stack.pop()
            for nxt in topology.get(cur, ()):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)

    text_count = 0
    try:
        from pdf_pipeline import pytesseract  # local optional dependency

        if pytesseract is not None:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            prep = cv2.adaptiveThreshold(
                cv2.GaussianBlur(gray, (3, 3), 0),
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11,
            )
            data = pytesseract.image_to_data(
                prep,
                output_type=pytesseract.Output.DICT,
                config="--oem 3 --psm 11",
            )
            confs = data.get("conf", [])
            words = data.get("text", [])
            text_count = sum(
                1
                for i, w in enumerate(words)
                if str(w).strip() and i < len(confs) and float(confs[i]) >= 35.0
            )
    except Exception:
        text_count = 0

    return {
        "line_count": int(line_count),
        "text_count": int(text_count),
        "junction_count": int(junction_count),
        "connected_regions": int(connected_regions),
        "total_path_length": int(total_path_length),
        "avg_segment_length": avg_segment_length,
        "line_density_per_mp": round(line_count / (roi_area_px / 1_000_000.0), 2),
        "junction_density_per_mp": round(junction_count / (roi_area_px / 1_000_000.0), 2),
        "roi_width_px": int(roi_w),
        "roi_height_px": int(roi_h),
    }


def _load_model_from_xml_string(xml_data: str):
    if ProteusSerializer is None:
        return None
    loader = ProteusSerializer()
    if hasattr(loader, "load_from_string"):
        return loader.load_from_string(xml_data)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False, encoding="utf-8"
    ) as f:
        f.write(xml_data)
        tmp = Path(f.name)
    try:
        return loader.load(tmp.parent, tmp.name)
    finally:
        tmp.unlink(missing_ok=True)


def render_xml_to_svg(xml_data: str) -> tuple[str | None, str | None]:
    """Return (svg_string, error_message). On failure both may be set partially for diagnostics."""
    def _svg_has_drawables(svg_text: str) -> bool:
        """Best-effort guard: reject renderer outputs that are structurally SVG but visually empty."""
        lower = svg_text.lower()
        drawable_tokens = (
            "<line",
            "<path",
            "<rect",
            "<circle",
            "<ellipse",
            "<polygon",
            "<polyline",
            "<text",
        )
        return any(tok in lower for tok in drawable_tokens)

    pydexpi_err: str | None = None
    model = None
    if ProteusSerializer is not None:
        try:
            model = _load_model_from_xml_string(xml_data)
        except Exception as exc:
            pydexpi_err = str(exc)
            model = None

    if model is not None and SvgRenderer is not None:
        try:
            svg = SvgRenderer().render(model)
            if _svg_has_drawables(svg):
                return svg, None
            pydexpi_err = pydexpi_err or (
                "pydexpi renderer returned an SVG with no drawable elements; "
                "using fallback renderer."
            )
        except Exception as exc:
            pydexpi_err = pydexpi_err or str(exc)

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, encoding="utf-8"
        ) as f:
            f.write(xml_data)
            tmp = Path(f.name)
        try:
            svg = render_fallback_svg(tmp)
            return svg, pydexpi_err
        finally:
            tmp.unlink(missing_ok=True)
    except Exception as exc:
        parts = [str(exc)]
        if pydexpi_err:
            parts.insert(0, f"pydexpi: {pydexpi_err}")
        return None, "; ".join(parts)


@app.route("/", methods=["GET", "POST"])
def upload_file():
    svg_content = None
    error = None
    note = None

    if request.method == "POST":
        file = request.files.get("xml_file")
        if not file or not file.filename:
            error = "Please choose an XML file."
        else:
            try:
                xml_data = file.read().decode("utf-8")
            except UnicodeDecodeError:
                error = "File must be UTF-8 text."
            else:
                svg_content, note = render_xml_to_svg(xml_data)
                if svg_content is None:
                    error = note or "Rendering failed."
                    note = None

    return render_template(
        "index.html", svg=svg_content, error=error, note=note
    )


@app.route("/api/graph/analyze", methods=["POST"])
def api_graph_analyze():
    """JSON: full graph pipeline for one or two uploaded XML diagrams."""
    file_a = request.files.get("xml_a")
    if not file_a or not file_a.filename:
        return jsonify({"error": "Upload XML A (required)."}), 400
    try:
        xml_a = file_a.read().decode("utf-8")
    except UnicodeDecodeError:
        return jsonify({"error": "XML A must be UTF-8 text."}), 400

    xml_b: str | None = None
    file_b = request.files.get("xml_b")
    if file_b and file_b.filename:
        try:
            xml_b = file_b.read().decode("utf-8")
        except UnicodeDecodeError:
            return jsonify({"error": "XML B must be UTF-8 text."}), 400

    try:
        result = run_full_pipeline(xml_a, xml_b)
    except ET.ParseError as exc:
        return jsonify({"error": f"Invalid XML: {exc}"}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


@app.route("/pdf-pipeline", methods=["GET"])
def pdf_pipeline_redirect():
    """Old URL: open home with PDF tab (hash picked up by index.html)."""
    return redirect(url_for("upload_file") + "#pdf-pipeline")


@app.route("/api/pdf-pipeline/process", methods=["POST"])
def api_pdf_pipeline_process():
    """Multipart: pdf_file, page   DPI is fixed to 300 for maximum fidelity."""
    f = request.files.get("pdf_file")
    if not f or not f.filename:
        return jsonify({"error": "Upload a PDF file."}), 400
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a .pdf."}), 400
    raw_page = request.form.get("page", "1").strip()
    try:
        page_num = int(raw_page)
    except ValueError:
        return jsonify({"error": "Page must be an integer  "}), 400
    dpi = 300

    try:
        pdf_bytes = f.read()
    except Exception as exc:
        return jsonify({"error": f"Read failed: {exc}"}), 400

    try:
        result = run_pdf_page_pipeline(pdf_bytes, page_num, dpi=dpi)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    xml_out = (result.get("step_xml") or {}).get("dexpi_like")
    if isinstance(xml_out, str) and xml_out.strip():
        svg_out, svg_note = render_xml_to_svg(xml_out)
        result["step_svg"] = {
            "rendered": bool(svg_out),
            "svg": svg_out,
            "note": svg_note,
        }
    else:
        result["step_svg"] = {
            "rendered": False,
            "svg": None,
            "note": "No DEXPI-like XML available to render.",
        }

    return jsonify(result)


@app.route("/api/pdf-svg/convert", methods=["POST"])
def api_pdf_svg_convert():
    """Multipart: pdf_file + page (1-based) -> direct visual SVG conversion."""
    f = request.files.get("pdf_file")
    if not f or not f.filename:
        return jsonify({"error": "Upload a PDF file."}), 400
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a .pdf."}), 400
    raw_page = request.form.get("page", "1").strip()
    try:
        page_num = int(raw_page)
    except ValueError:
        return jsonify({"error": "Page must be an integer (1-based)."}), 400
    try:
        pdf_bytes = f.read()
    except Exception as exc:
        return jsonify({"error": f"Read failed: {exc}"}), 400
    try:
        svg, meta = pdf_page_to_svg_string(pdf_bytes, page_num)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"pdf_meta": meta, "svg": svg})


@app.route("/api/pdf-svg/interpret", methods=["POST"])
def api_pdf_svg_interpret():
    """
    Multipart: pdf_file + page (1-based) -> visual SVG + level-1 visual analytics.
    """
    f = request.files.get("pdf_file")
    if not f or not f.filename:
        return jsonify({"error": "Upload a PDF file."}), 400
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a .pdf."}), 400
    raw_page = request.form.get("page", "1").strip()
    try:
        page_num = int(raw_page)
    except ValueError:
        return jsonify({"error": "Page must be an integer (1-based)."}), 400
    try:
        pdf_bytes = f.read()
    except Exception as exc:
        return jsonify({"error": f"Read failed: {exc}"}), 400

    try:
        svg, meta = pdf_page_to_svg_string(pdf_bytes, page_num)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    svg_metrics = _estimate_visual_analytics_from_svg(
        svg,
        float(meta.get("width_pt") or 0.0),
        float(meta.get("height_pt") or 0.0),
    )
    image_metrics = _estimate_visual_metrics_from_image(pdf_bytes, page_num, dpi=300)
    metrics = {
        "line_count": int(image_metrics.get("line_count", 0)),
        "text_count": int(image_metrics.get("text_count", 0)),
        "junction_count": int(image_metrics.get("junction_count", 0)),
        "connected_regions": int(image_metrics.get("connected_regions", 0)),
        "total_path_length": int(image_metrics.get("total_path_length", 0)),
        "avg_segment_length": float(image_metrics.get("avg_segment_length", 0.0)),
        "line_density_per_mp": float(image_metrics.get("line_density_per_mp", 0.0)),
        "junction_density_per_mp": float(image_metrics.get("junction_density_per_mp", 0.0)),
        "roi_width_px": int(image_metrics.get("roi_width_px", 0)),
        "roi_height_px": int(image_metrics.get("roi_height_px", 0)),
        "svg_line_elements": int(svg_metrics.get("line_element_count", 0)),
        "svg_path_elements": int(svg_metrics.get("path_count", 0)),
    }
    complexity_index = round(
        (
            min(metrics["line_count"] / 30.0, 40.0)
            + min(metrics["junction_count"] * 0.35, 25.0)
            + min(metrics["connected_regions"] * 0.8, 15.0)
            + min(metrics["text_count"] / 20.0, 20.0)
        ),
        1,
    )

    return jsonify(
        {
            "pdf_meta": meta,
            "svg": svg,
            "metrics": metrics,
            "complexity_index": complexity_index,
            "metric_sources": {
                "primary": "image",
                "image_metrics": image_metrics,
                "svg_metrics": svg_metrics,
            },
            "insights": {
                "fragmentation_hint": (
                    "High fragmentation" if int(metrics["connected_regions"]) >= 12 else "Moderate/low fragmentation"
                ),
                "complexity_hint": (
                    "High visual complexity" if complexity_index >= 70 else "Moderate visual complexity"
                ),
            },
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
