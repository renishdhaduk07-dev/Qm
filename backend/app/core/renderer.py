"""
Backend quilt renderer.

Renders a validated quilt dict into a PNG image using Matplotlib.
Each section is drawn as a plain filled polygon with a white border.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")  # headless backend for server-side rendering
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


# ---------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------
def render_quilt_to_png(quilt: dict) -> str:
    """Render a validated quilt dict into a base64-encoded PNG string.

    Args:
        quilt: A validated quilt dict with sections, swatches, etc.

    Returns:
        Base64-encoded PNG image string.
    """
    w = quilt["size"]["width"]
    h = quilt["size"]["height"]

    # Build swatch lookup
    swatch_map = {
        sw["id"]: sw.get("materialColor", "#CCCCCC")
        for sw in quilt.get("swatches", [])
    }

    fig, ax = plt.subplots(
        1, 1,
        figsize=(10, 10 * h / max(w, 1)),
        facecolor="#1e1e1e",
        dpi=120,
    )
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_facecolor("#1e1e1e")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("white")

    for section in quilt.get("sections", []):
        coords = section["polygon"]["geometry"]["coordinates"][0]
        base_color = swatch_map.get(section["swatchId"], "#CCCCCC")

        ax.add_patch(MplPolygon(
            coords, closed=True, linewidth=0.8,
            edgecolor="white", facecolor=base_color,
        ))

    ax.set_title(
        quilt.get("name", "Quilt"),
        color="white", fontsize=14, fontweight="bold", pad=12,
    )

    # Render to PNG bytes → base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
