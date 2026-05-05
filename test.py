from pathlib import Path
import xml.etree.ElementTree as ET


def _local_name(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _iter_local(root: ET.Element, name: str):
    for el in root.iter():
        if _local_name(el.tag) == name:
            yield el


try:
    from pydexpi.loaders import ProteusSerializer  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    ProteusSerializer = None

try:
    from pydexpi.renderers import SvgRenderer  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    SvgRenderer = None


def render_fallback_svg(xml_path: Path) -> str:
    """Render a minimal SVG from Equipment/Position and Connection tags."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    equipment_positions: dict[str, tuple[float, float, str]] = {}
    for equipment in _iter_local(root, "Equipment"):
        equipment_id = equipment.get("id")
        if not equipment_id:
            continue
        position = next(
            (c for c in equipment if _local_name(c.tag) == "Position"), None
        )
        if position is None:
            continue
        x = float(position.get("x", "0"))
        y = float(position.get("y", "0"))
        label = equipment.get("name", equipment_id)
        equipment_positions[equipment_id] = (x, y, label)

    if equipment_positions:
        xs = [p[0] for p in equipment_positions.values()]
        ys = [p[1] for p in equipment_positions.values()]
        min_x, max_x = min(xs) - 80, max(xs) + 80
        min_y, max_y = min(ys) - 80, max(ys) + 80
    else:
        min_x, max_x, min_y, max_y = 0, 600, 0, 300

    w, h = max_x - min_x, max_y - min_y
    lines: list[str] = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="{min_x} {min_y} {w} {h}">'
        ),
        f'<rect x="{min_x}" y="{min_y}" width="{w}" height="{h}" fill="white"/>',
    ]

    for connection in _iter_local(root, "Connection"):
        from_id = connection.get("from")
        to_id = connection.get("to")
        if not from_id or not to_id:
            continue
        if from_id not in equipment_positions or to_id not in equipment_positions:
            continue
        x1, y1, _ = equipment_positions[from_id]
        x2, y2, _ = equipment_positions[to_id]
        lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            'stroke="#2563eb" stroke-width="4" />'
        )

    for _, (x, y, label) in equipment_positions.items():
        lines.append(
            f'<rect x="{x - 35}" y="{y - 20}" width="70" height="40" '
            'rx="6" fill="#f3f4f6" stroke="#111827" />'
        )
        lines.append(
            f'<text x="{x}" y="{y + 5}" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="12" fill="#111827">'
            f"{label}</text>"
        )

    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    xml_path = Path("sample.xml")
    output_svg_path = Path("output.svg")
    if not xml_path.exists():
        print(f"Missing input XML: {xml_path.resolve()}")
        return

    print(f"Loading XML: {xml_path.resolve()}")
    model = None

    if ProteusSerializer is None:
        print("pydexpi not available in active environment; using fallback renderer.")
    else:
        loader = ProteusSerializer()
        try:
            model = loader.load(xml_path.parent, xml_path.name)
            print("\n[MODEL LOADED]")
            print(model)
        except Exception as exc:
            print("\n[LOAD FAILED]")
            print(f"Error: {exc}")
            print("Continuing with fallback renderer.")

    print("\n[RENDERING]")
    if model is not None and SvgRenderer is not None:
        try:
            renderer = SvgRenderer()
            svg = renderer.render(model)
            output_svg_path.write_text(svg, encoding="utf-8")
            print(f"SVG written to: {output_svg_path.resolve()} (pydexpi renderer)")
            return
        except Exception as exc:
            print(f"pydexpi renderer failed: {exc}")
            print("Falling back to minimal XML-based renderer.")

    svg = render_fallback_svg(xml_path)
    output_svg_path.write_text(svg, encoding="utf-8")
    print(f"SVG written to: {output_svg_path.resolve()} (fallback renderer)")


if __name__ == "__main__":
    main()
