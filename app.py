"""Minimal Flask POC: upload DEXPI/Proteus XML, render SVG in memory, show in browser."""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for

from pid_graph import run_full_pipeline
from pdf_pipeline import pdf_page_to_svg_string, run_pdf_page_pipeline
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
    Multipart: pdf_file + page (1-based) -> visual SVG + extracted object graph.
    Uses non-strict quality mode so results are returned for inspection.
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
        parsed = run_pdf_page_pipeline(
            pdf_bytes,
            page_num,
            dpi=300,
            strict_quality=False,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    step_graph = parsed.get("step_graph") or {}
    obj = {
        "nodes": [
            {
                "id": n.get("id"),
                "label": n.get("name") or n.get("id"),
                "type": n.get("type"),
                "position": n.get("position"),
            }
            for n in step_graph.get("nodes", [])
        ],
        "edges": [
            {
                "from": e.get("from"),
                "to": e.get("to"),
                "type": "connection",
                "id": e.get("piping_component_id"),
            }
            for e in step_graph.get("edges", [])
        ],
        "quality": ((parsed.get("step_debug") or {}).get("quality") or {}),
    }

    return jsonify(
        {
            "pdf_meta": meta,
            "svg": svg,
            "parsed_object": obj,
            "simple_graph_svg": ((parsed.get("step_simple_graph_svg") or {}).get("svg")),
            "debug_overlay_base64": ((parsed.get("step_debug") or {}).get("overlay_base64")),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
