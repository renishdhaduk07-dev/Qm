"""Quilt geometry validator.

Uses **Pydantic** as the authoritative schema validator (single source of
truth) and **Shapely** for geometry checks.  Returns a structured
``ValidationResult`` with per-section error details so the retry prompt
builder can produce *surgical* fix instructions.

Validation pipeline:
  1. Pydantic schema parse (``QuiltLayout.model_validate``)
  2. Size match
  3. Swatch cross-references
  4. Per-section geometry
     a. Single-ring rule
     b. Ring closure
     c. Coordinate bounds
     d. Self-intersection / validity
     e. Sliver detection (near-zero area)
     f. Tiny-edge detection
  5. Pairwise overlap — **STRtree** spatial index (O(n log n) average)
  6. Full canvas coverage
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Set

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.strtree import STRtree
from shapely.validation import make_valid

from app.core.models import QuiltLayout

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
GRID_STEP = 0.5  # snap-to-grid resolution
_MIN_AREA_RATIO = 1e-4  # min section area as fraction of canvas area
_MIN_EDGE_RATIO = 1e-3  # min edge length as fraction of min(w, h)
_BASE_AREA_TOL = 0.5  # base tolerance for area comparisons


# ---------------------------------------------------------------
# Structured validation result
# ---------------------------------------------------------------
@dataclass
class ValidationError:
    """A single validation failure."""

    error_type: str  # schema | size | swatch_ref | ring |
    #                  closure | bounds | self_intersection |
    #                  empty | sliver | tiny_edge | overlap | coverage
    message: str
    section_id: Optional[str] = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict[str, object] = {
            "error_type": self.error_type,
            "message": self.message,
        }
        if self.section_id:
            d["section_id"] = self.section_id
        if self.details:
            d["details"] = self.details
        return d


@dataclass
class ValidationResult:
    """Aggregated outcome of all validation checks."""

    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)

    # -- helpers for the retry prompt builder -----------------------

    def summary(self) -> str:
        """Human-readable error summary (one line per error)."""
        if self.is_valid:
            return "Valid"
        lines = []
        for e in self.errors:
            prefix = f"[{e.section_id}] " if e.section_id else ""
            lines.append(f"- {prefix}{e.error_type}: {e.message}")
        return "\n".join(lines)

    def failing_section_ids(self) -> Set[str]:
        """Set of section IDs that have at least one error."""
        return {e.section_id for e in self.errors if e.section_id}

    def errors_by_type(self) -> dict:
        result: dict[str, list] = {}
        for e in self.errors:
            result.setdefault(e.error_type, []).append(e)
        return result

    def to_dict_list(self) -> list:
        """Serialisable list of error dicts (for LangGraph state)."""
        return [e.to_dict() for e in self.errors]


# ---------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------
def _snap(value: float, step: float) -> float:
    """Snap a single coordinate to the nearest grid step."""
    return round(round(value / step) * step, 6)


def normalize_quilt(quilt: dict, grid_step: float = GRID_STEP) -> dict:
    """Normalise quilt geometry **in-place**.

    * Snaps every coordinate to the nearest *grid_step*.
    * Ensures every polygon ring is closed (first == last).
    """
    for section in quilt.get("sections", []):
        coord_rings = (
            section.get("polygon", {})
            .get("geometry", {})
            .get("coordinates", [])
        )
        for idx, ring in enumerate(coord_rings):
            snapped = [
                [_snap(pt[0], grid_step), _snap(pt[1], grid_step)]
                for pt in ring
            ]
            # Ensure closure
            if snapped and snapped[0] != snapped[-1]:
                snapped.append(list(snapped[0]))
            coord_rings[idx] = snapped
    return quilt


# ---------------------------------------------------------------
# Edge-length helper
# ---------------------------------------------------------------
def _edge_length(a, b) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------
def validate_quilt(
    quilt: dict, width: int, height: int
) -> ValidationResult:
    """Validate a quilt dict and return structured results.

    Returns a ``ValidationResult`` whose ``.is_valid`` is True only when
    **every** check passes.  On failure, ``.errors`` contains one entry
    per problem, each tagged with the offending ``section_id`` where
    applicable.
    """
    errors: List[ValidationError] = []

    # ── 1. Pydantic schema validation (authoritative) ─────────────
    try:
        QuiltLayout.model_validate(quilt)
    except Exception as exc:
        errors.append(
            ValidationError(
                error_type="schema",
                message=f"Pydantic validation failed: {exc}",
            )
        )
        # Schema failure is fatal — can't trust the rest of the dict
        return ValidationResult(is_valid=False, errors=errors)

    # ── 2. Size match ─────────────────────────────────────────────
    qw, qh = quilt["size"]["width"], quilt["size"]["height"]
    if qw != width or qh != height:
        errors.append(
            ValidationError(
                error_type="size",
                message=f"Expected {width}x{height}, got {qw}x{qh}.",
            )
        )

    sections = quilt.get("sections", [])
    if not sections:
        errors.append(
            ValidationError(error_type="schema", message="No sections.")
        )
        return ValidationResult(is_valid=False, errors=errors)

    # ── 3. Swatch cross-references ────────────────────────────────
    swatch_ids = {s["id"] for s in quilt.get("swatches", [])}

    # ── Derived thresholds ────────────────────────────────────────
    expected_area = width * height
    area_tol = max(_BASE_AREA_TOL, expected_area * 0.0001)
    min_area = expected_area * _MIN_AREA_RATIO
    min_edge = min(width, height) * _MIN_EDGE_RATIO

    shapely_polys: list[tuple[str, ShapelyPolygon]] = []

    # ── 5. Per-section checks ─────────────────────────────────────
    for section in sections:
        sid = section["id"]

        # 5a. Swatch ref
        if section["swatchId"] not in swatch_ids:
            errors.append(
                ValidationError(
                    error_type="swatch_ref",
                    message=f"Unknown swatchId '{section['swatchId']}'.",
                    section_id=sid,
                )
            )

        coord_rings = section["polygon"]["geometry"]["coordinates"]

        # 5b. Single ring
        if len(coord_rings) != 1:
            errors.append(
                ValidationError(
                    error_type="ring",
                    message=f"Has {len(coord_rings)} rings; expected 1.",
                    section_id=sid,
                )
            )
            continue

        ring = coord_rings[0]

        # 5c. Closure
        if ring[0] != ring[-1]:
            errors.append(
                ValidationError(
                    error_type="closure",
                    message=f"Ring not closed: first={ring[0]}, last={ring[-1]}.",
                    section_id=sid,
                )
            )

        # 5d. Bounds
        for coord in ring:
            x, y = coord[0], coord[1]
            if not (0 <= x <= width and 0 <= y <= height):
                errors.append(
                    ValidationError(
                        error_type="bounds",
                        message=(
                            f"Out-of-bounds ({x}, {y}); "
                            f"must be in 0..{width}, 0..{height}."
                        ),
                        section_id=sid,
                        details={"x": x, "y": y},
                    )
                )
                break  # one bounds error per section is enough

        # Build Shapely polygon
        try:
            poly = ShapelyPolygon(ring)

            # 5e. Self-intersection / validity
            if not poly.is_valid:
                errors.append(
                    ValidationError(
                        error_type="self_intersection",
                        message="Polygon is self-intersecting or invalid.",
                        section_id=sid,
                    )
                )
                repaired = make_valid(poly)
                if not isinstance(repaired, ShapelyPolygon):
                    errors.append(
                        ValidationError(
                            error_type="geometry",
                            message="Polygon could not be repaired to a simple Polygon.",
                            section_id=sid,
                        )
                    )
                    continue
                poly = repaired

            if poly.is_empty:
                errors.append(
                    ValidationError(
                        error_type="empty",
                        message="Polygon is empty after validation.",
                        section_id=sid,
                    )
                )
                continue

            # 5f. Sliver (near-zero area)
            if poly.area < min_area:
                errors.append(
                    ValidationError(
                        error_type="sliver",
                        message=f"Area {poly.area:.4f} < minimum {min_area:.4f}.",
                        section_id=sid,
                        details={
                            "area": round(poly.area, 4),
                            "min_area": round(min_area, 4),
                        },
                    )
                )

            # 5g. Tiny edges
            ext_coords = list(poly.exterior.coords)
            for i in range(len(ext_coords) - 1):
                elen = _edge_length(ext_coords[i], ext_coords[i + 1])
                if elen < min_edge:
                    errors.append(
                        ValidationError(
                            error_type="tiny_edge",
                            message=(
                                f"Edge length {elen:.4f} < "
                                f"minimum {min_edge:.4f}."
                            ),
                            section_id=sid,
                            details={"edge_length": round(elen, 4)},
                        )
                    )
                    break  # one tiny-edge error per section

            shapely_polys.append((sid, poly))

        except Exception as exc:
            errors.append(
                ValidationError(
                    error_type="geometry",
                    message=f"Invalid geometry: {exc}",
                    section_id=sid,
                )
            )

    # ── 6. Overlap detection (STRtree spatial index) ──────────────
    if len(shapely_polys) > 1:
        polys_only = [p for _, p in shapely_polys]
        tree = STRtree(polys_only)

        checked: set[tuple[int, int]] = set()
        for i, (sid_a, poly_a) in enumerate(shapely_polys):
            candidate_indices = tree.query(poly_a)
            for j_idx in candidate_indices:
                j = int(j_idx)
                if j <= i:
                    continue
                pair = (i, j)
                if pair in checked:
                    continue
                checked.add(pair)

                sid_b, poly_b = shapely_polys[j]
                inter = poly_a.intersection(poly_b)
                if inter.area > area_tol:
                    errors.append(
                        ValidationError(
                            error_type="overlap",
                            message=(
                                f"Overlaps with '{sid_b}' "
                                f"(intersection area={inter.area:.4f})."
                            ),
                            section_id=sid_a,
                            details={
                                "other_section": sid_b,
                                "overlap_area": round(inter.area, 4),
                            },
                        )
                    )

    # ── 7. Full coverage ──────────────────────────────────────────
    if shapely_polys:
        total_area = sum(p.area for _, p in shapely_polys)
        if abs(total_area - expected_area) > area_tol:
            errors.append(
                ValidationError(
                    error_type="coverage",
                    message=(
                        f"Total area={total_area:.4f}; "
                        f"expected={expected_area}."
                    ),
                    details={
                        "total_area": round(total_area, 4),
                        "expected_area": expected_area,
                    },
                )
            )

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)
