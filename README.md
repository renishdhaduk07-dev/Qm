# AI Quilt Generator

An AI-powered quilt pattern generator using **LangGraph** for workflow orchestration, **Google Gemini 2.5 Flash** (via structured output / tool calling) for pattern creation, **Shapely** for geometry validation, and **Matplotlib** for server-side rendering.

## Architecture

```
START → Prompt Builder → Gemini Generator → Validator ─┬─→ Renderer → END
                              ↑                         │
                              └──── surgical retry ◄────┘
                              (escalating constraints, max 3 attempts)
```

### Key Design Decisions

- **Pydantic as single source of truth** — the `QuiltLayout` model defines the schema; JSON Schema and prompt constraints are auto-derived from it.
- **Structured output (Tool Calling)** — Gemini returns a validated Pydantic object directly, eliminating fragile JSON parsing.
- **Surgical retry** — on validation failure, only failing section IDs are fed back with escalating constraints (attempt 2: snap to integers; attempt 3: simplify to rectangles).
- **STRtree spatial indexing** — pairwise overlap detection runs in O(n log n) instead of O(n²).
- **Server-side rendering** — Matplotlib renders the quilt to PNG on the backend; the frontend displays the image without needing plotting libraries.

## Quilt Styles

| Style | Key |
|---|---|
| 🧩 Classic Patchwork | `classic_patchwork` |
| ⭐ Star Burst | `star_burst` |
| 🏠 Log Cabin | `log_cabin` |
| 🎱 Snowball / Octagon | `snowball` |
| 🚧 Rail Fence | `rail_fence` |
| 🪚 Herringbone / Chevron | `herringbone` |
| ♟️ Checkerboard | `checkerboard` |

## Workflow

The entire generation pipeline is orchestrated as a **LangGraph `StateGraph`** with four nodes and a conditional retry edge. The graph is compiled once at module load and reused for every request.

### State

All nodes read from and write to a shared `QuiltState` dict:

| Key | Type | Purpose |
|---|---|---|
| `width`, `height` | int | Requested canvas dimensions |
| `style` | str | Pattern style key |
| `system_prompt` | str | Built once by the prompt node |
| `prompt` | str | Per-request user prompt (or retry prompt) |
| `quilt` | dict | Gemini's parsed output (`QuiltLayout`) |
| `image` | str | Base64-encoded PNG from the renderer |
| `error` | str | Last validation/generation error message |
| `validation_errors` | list | Structured error dicts for surgical retry |
| `attempts` | int | Current attempt counter (max 3) |

### Nodes

#### 1. Prompt Builder
- Calls `build_system_prompt()` — assembles the system prompt with schema constraints (auto-derived from Pydantic), GeoJSON rules, and design principles.
- Calls `build_prompt(width, height, style)` — selects the per-style strategy (geometry examples, color palette, layout approach) and injects canvas dimensions.
- Writes `system_prompt` and `prompt` to state. Does **not** call the LLM.

#### 2. Gemini Generator
- On **first attempt**: sends `system_prompt` + `prompt` to Gemini 2.5 Flash via LangChain structured output (`llm.with_structured_output(QuiltLayout)`).
- On **retry**: builds a **surgical retry prompt** by appending to the original prompt:
  - The exact failing section IDs and per-error details (type + message).
  - An instruction to modify *only* the broken sections and keep everything else unchanged.
  - **Escalating constraints** based on attempt number:

  | Attempt | Strategy |
  |---|---|
  | 1 | Targeted fixes — only the failing sections are flagged |
  | 2 | Snap all coordinates to integers, restrict to rectangles + triangles, enforce ring closure |
  | 3 | Maximum simplification — reduce section count by 30%, rectangles only, grid-based layout |

- Gemini returns a validated `QuiltLayout` Pydantic object; this is dumped to a dict and stored as `quilt`.

#### 3. Validator
- **Normalizes** geometry first: snaps coordinates to a 0.5 grid and auto-closes open polygon rings.
- Runs the full **6-step validation pipeline** (see [Validation Pipeline](#validation-pipeline) below).
- If valid → clears `error`, flow proceeds to the renderer.
- If invalid → sets `error` with a human-readable summary and `validation_errors` with structured dicts (each containing `error_type`, `message`, `section_id`, `details`). Clears `quilt` so the retry path triggers.

#### 4. Renderer
- Takes the validated quilt dict and renders it to a PNG using **Matplotlib** (headless `Agg` backend).
- Each section is drawn as a filled `MplPolygon` with a white border; swatch colors are looked up by ID.
- The figure is saved to an in-memory buffer, base64-encoded, and stored as `image` in state.

### Conditional Edge (Retry Logic)

After the Validator, a `should_retry` function decides the next step:

```
if error exists AND attempts < 3 → route back to Gemini Generator ("retry")
otherwise                         → route to Renderer ("end")
```

This creates the retry loop shown in the architecture diagram. If all 3 attempts fail, the flow still reaches the renderer (which returns `image: None`), and `run_quilt_flow()` raises a `RuntimeError`.

### Graph Edges

```
prompt_builder ──→ gemini_generator ──→ validator ──┬──→ renderer ──→ END
                        ↑                           │
                        └───── retry (conditional) ◄┘
```

## Setup

### 1. Environment

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your-actual-api-key
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Activate it:

- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

You should see `(.venv)` in your terminal prompt once activated.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies: FastAPI, Uvicorn, LangGraph, LangChain-Google-GenAI, Shapely, Pydantic v2, python-dotenv, Streamlit, Matplotlib, Requests.

### 4. Start the backend

```bash
cd backend
uvicorn app.main:app --reload
```

API available at `http://localhost:8000`.

### 5. Start the frontend

In a separate terminal (with the venv activated):

```bash
cd frontend
streamlit run app.py
```

Streamlit app opens at `http://localhost:8501`.

## API

### `POST /generate-quilt`

**Request:**
```json
{
  "width": 90,
  "height": 108,
  "style": "star_burst"
}
```

| Field | Type | Description |
|---|---|---|
| `width` | float | Canvas width in inches (> 0) |
| `height` | float | Canvas height in inches (> 0) |
| `style` | string | Pattern style key (default: `classic_patchwork`) |

**Response:** A complete quilt JSON object containing:
- `id` — unique quilt UUID
- `name` — AI-generated creative name
- `size` — `{ width, height }`
- `sections[]` — GeoJSON polygon features with swatch references
- `swatches[]` — color definitions (`{ id, materialType, materialColor }`)
- `image` — base64-encoded PNG rendering

### `GET /health`

Returns `{ "status": "ok" }`.

## Validation Pipeline

The validator runs on every Gemini response before rendering:

1. **Pydantic schema parse** — `QuiltLayout.model_validate()`
2. **Size match** — canvas dimensions match the request
3. **Swatch cross-references** — every `section.swatchId` maps to a valid swatch
4. **Per-section geometry** — single-ring rule, ring closure, coordinate bounds, self-intersection, sliver detection, tiny-edge detection
5. **Pairwise overlap** — STRtree spatial index for efficient O(n log n) detection
6. **Full canvas coverage** — section union must cover ≥ 95% of the canvas area

Geometry is normalized before validation (snap-to-grid at 0.5 resolution, auto-close rings).

## Project Structure

```
ai-quilt-generator/
├── backend/
│   └── app/
│       ├── main.py                # FastAPI app entrypoint
│       ├── api/
│       │   └── quilt.py           # POST /generate-quilt endpoint
│       ├── core/
│       │   ├── models.py          # Pydantic models (single source of truth)
│       │   ├── langgraph_flow.py  # LangGraph StateGraph + retry logic
│       │   ├── prompt.py          # System & per-style user prompts
│       │   ├── validator.py       # Pydantic + Shapely geometry validation
│       │   └── renderer.py        # Matplotlib PNG renderer
│       └── services/
│           └── gemini_service.py  # Gemini 2.5 Flash structured output wrapper
├── frontend/
│   └── app.py                     # Streamlit UI (size presets, style picker, display)
├── quilt.schema.json              # Auto-generated JSON Schema from Pydantic
├── requirements.txt               # Python dependencies
├── .env                           # GEMINI_API_KEY (not committed)
└── README.md
```

## Data Model

```
QuiltLayout
├── id: str (UUID)
├── name: str
├── size: { width: int, height: int }
├── sections[]:
│   ├── id: str (UUID)
│   ├── type: str (rectangle | triangle | diamond | hexagon | polygon)
│   ├── polygon: GeoJSON Feature { type: "Feature", geometry: { type: "Polygon", coordinates } }
│   └── swatchId: str → references Swatch.id
└── swatches[]:
    ├── id: str (UUID)
    ├── materialType: "color"
    └── materialColor: str (hex, e.g. #FF5733)
```
