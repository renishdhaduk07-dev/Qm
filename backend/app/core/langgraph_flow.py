"""
LangGraph workflow for the AI Quilt Generator.

Defines a StateGraph with four nodes (prompt_builder, gemini_generator,
validator, renderer) and a conditional retry edge.
"""

from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.core.prompt import build_prompt, build_system_prompt
from app.core.renderer import render_quilt_to_png
from app.core.validator import normalize_quilt, validate_quilt
from app.services.gemini_service import call_gemini

MAX_ATTEMPTS = 3


# ---------------------------------------------------------------
# State definition
# ---------------------------------------------------------------
class QuiltState(TypedDict):
    width: int
    height: int
    style: str
    system_prompt: Optional[str]
    prompt: Optional[str]
    quilt: Optional[dict]
    image: Optional[str]  # base64-encoded PNG
    error: Optional[str]
    validation_errors: Optional[list]  # structured error dicts for retry
    attempts: int


# ---------------------------------------------------------------
# Retry prompt builder (surgical fixes + escalation)
# ---------------------------------------------------------------
def _build_retry_prompt(
    original_prompt: str,
    validation_errors: list,
    attempt: int,
    width: int,
    height: int,
) -> str:
    """Build a surgical retry prompt with escalating constraints.

    Attempt 1 -> targeted fixes on failing sections only.
    Attempt 2 -> force grid-aligned integers, restrict to simple shapes.
    Attempt 3 -> maximum simplification (fewer sections, rectangles only).
    """
    failing_ids = sorted(
        {e["section_id"] for e in validation_errors if e.get("section_id")}
    )

    error_lines = []
    for e in validation_errors:
        prefix = (
            f"[{e['section_id']}] " if e.get("section_id") else "[GLOBAL] "
        )
        error_lines.append(f"  - {prefix}{e['error_type']}: {e['message']}")
    error_block = "\n".join(error_lines) or "  (no structured details)"

    parts = [
        original_prompt,
        "",
        f"== REPAIR MODE -- Attempt {attempt} of {MAX_ATTEMPTS} ==",
        "Your previous layout FAILED validation.",
        "",
        "FAILED CHECKS:",
        error_block,
    ]

    if failing_ids:
        ids_str = ", ".join(failing_ids)
        parts += [
            "",
            f"FAILING SECTION IDs: {ids_str}",
            "CRITICAL: Only modify the sections listed above. Keep ALL other "
            "sections EXACTLY as they were (same ids, coordinates, colors).",
        ]

    if attempt == 2:
        parts += [
            "",
            "ADDITIONAL CONSTRAINTS (tighter rules -- attempt 2):",
            "* Snap ALL coordinates to INTEGER values (whole numbers only).",
            "* Use ONLY rectangles and triangles (no complex polygons).",
            "* Ensure every polygon ring is closed (first point == last point).",
            "* Double-check that no two sections overlap.",
        ]
    elif attempt >= 3:
        parts += [
            "",
            "ADDITIONAL CONSTRAINTS (maximum simplification -- attempt 3):",
            "* REDUCE total section count by ~30%.",
            '* Use ONLY axis-aligned rectangles (type="rectangle").',
            "* ALL coordinates must be integers.",
            "* Use a simple grid-based layout to guarantee zero gaps/overlaps.",
            f"* Canvas is {width} wide x {height} tall. "
            f"Total area = {width * height}.",
        ]

    parts += [
        "",
        "Fix reminders:",
        "* BAD ring (not closed):  [[0,0],[10,0],[10,10],[0,10]]",
        "* GOOD ring (closed):     [[0,0],[10,0],[10,10],[0,10],[0,0]]",
        "* Every coordinate must satisfy 0 <= x <= width and 0 <= y <= height.",
        "",
        "Return the COMPLETE corrected JSON layout.",
    ]

    return "\n".join(parts)


# ---------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------
def prompt_builder_node(state: QuiltState) -> dict[str, Any]:
    """Build the system + user prompts from width/height/style. Does NOT call the LLM."""
    system_prompt = build_system_prompt()
    user_prompt = build_prompt(state["width"], state["height"], state["style"])
    return {"system_prompt": system_prompt, "prompt": user_prompt, "error": None}


def gemini_generator_node(state: QuiltState) -> dict[str, Any]:
    """Call Gemini and parse JSON. On retry, builds a surgical prompt
    with exact failing section IDs and escalating constraints."""
    attempts = state.get("attempts", 0) + 1
    system_prompt = state["system_prompt"]
    user_prompt = state["prompt"]

    # If retrying, build a surgical retry prompt with escalation
    if state.get("error"):
        user_prompt = _build_retry_prompt(
            original_prompt=user_prompt,
            validation_errors=state.get("validation_errors") or [],
            attempt=attempts,
            width=state["width"],
            height=state["height"],
        )

    try:
        quilt = call_gemini(system_prompt, user_prompt)
        return {
            "quilt": quilt,
            "error": None,
            "validation_errors": None,
            "attempts": attempts,
        }
    except (ValueError, RuntimeError) as e:
        return {"quilt": None, "error": str(e), "attempts": attempts}


def validator_node(state: QuiltState) -> dict[str, Any]:
    """Normalize geometry, then validate with structured error reporting."""
    quilt = state.get("quilt")
    if quilt is None:
        return {
            "error": state.get("error") or "No quilt generated.",
            "validation_errors": None,
        }

    # Normalize geometry (snap-to-grid, close rings) before validation
    normalize_quilt(quilt)

    result = validate_quilt(quilt, state["width"], state["height"])

    if result.is_valid:
        return {"error": None, "validation_errors": None}
    else:
        return {
            "quilt": None,
            "image": None,
            "error": result.summary(),
            "validation_errors": result.to_dict_list(),
        }


def render_node(state: QuiltState) -> dict[str, Any]:
    """Render the validated quilt into a PNG image on the backend."""
    quilt = state.get("quilt")
    print(f"[RENDER_NODE] Called. quilt={'present' if quilt else 'None'}")
    if quilt is None:
        return {"image": None}
    try:
        image_b64 = render_quilt_to_png(quilt)
        print(f"[RENDER_NODE] Image generated, length={len(image_b64)}")
        return {"image": image_b64}
    except Exception as e:
        print(f"[RENDER_NODE] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"image": None, "error": f"Rendering failed: {e}"}


# ---------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------
def should_retry(state: QuiltState) -> str:
    """Decide whether to retry generation or finish."""
    if state.get("error") and state.get("attempts", 0) < MAX_ATTEMPTS:
        return "retry"
    return "end"


# ---------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------
def _build_graph() -> StateGraph:
    """Construct the LangGraph StateGraph."""
    graph = StateGraph(QuiltState)

    # Add nodes
    graph.add_node("prompt_builder", prompt_builder_node)
    graph.add_node("gemini_generator", gemini_generator_node)
    graph.add_node("validator", validator_node)
    graph.add_node("renderer", render_node)

    # Define edges
    graph.set_entry_point("prompt_builder")
    graph.add_edge("prompt_builder", "gemini_generator")
    graph.add_edge("gemini_generator", "validator")

    # Conditional edge from validator
    graph.add_conditional_edges(
        "validator",
        should_retry,
        {
            "retry": "gemini_generator",
            "end": "renderer",  # render after validation passes
        },
    )

    # Renderer → END
    graph.add_edge("renderer", END)

    return graph


# Compile once at module level
_compiled_graph = _build_graph().compile()


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------
def run_quilt_flow(
    width: int,
    height: int,
    style: str = "classic_patchwork",
) -> dict:
    """Execute the full quilt generation workflow.

    Args:
        width: Canvas width.
        height: Canvas height.
        style: Pattern style key.

    Returns:
        Valid quilt dict.

    Raises:
        RuntimeError: If all retry attempts are exhausted without
        producing a valid quilt.
    """
    initial_state: QuiltState = {
        "width": width,
        "height": height,
        "style": style,
        "system_prompt": None,
        "prompt": None,
        "quilt": None,
        "image": None,
        "error": None,
        "validation_errors": None,
        "attempts": 0,
    }

    result = _compiled_graph.invoke(initial_state)

    if result.get("quilt") is None:
        raise RuntimeError(
            f"Quilt generation failed after {result.get('attempts', 0)} "
            f"attempts. Last error: {result.get('error', 'Unknown error')}"
        )

    quilt = result["quilt"]
    quilt["image"] = result.get("image")
    return quilt
