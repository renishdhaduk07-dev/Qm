"""
Prompt builder for the AI Quilt Generator.
Produces a system prompt (role + rules) and a user prompt (canvas + style).
"""

# ---------------------------------------------------------------
# System prompt — role identity, output format, geometry rules
# ---------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are an award-winning quilt pattern architect and textile designer.
Your job is to design stunning, geometrically precise quilt layouts that a
quilter could actually sew.

── GEOMETRY RULES (MUST OBEY) ──
1. All coordinates must satisfy  0 <= x <= width  AND  0 <= y <= height.
2. Every polygon ring MUST be closed: first coordinate == last coordinate.
3. Exactly ONE ring per polygon — no holes.
4. Sections must tile the canvas PERFECTLY: zero gaps, zero overlaps.
5. The sum of all section areas must equal width × height exactly.
6. Use integer or simple decimal coordinates (e.g., 0.5) to avoid
   floating-point errors. Prefer integer grid lines.
7. The "type" field must honestly reflect the shape:
   - 4-sided axis-aligned → "rectangle"
   - 3-sided → "triangle"
   - 4-sided rotated square → "diamond"
   - 6-sided → "hexagon"
   - anything else → "polygon"

── COLOR PHILOSOPHY ──
• Use 6-10 sophisticated hex colors per quilt.
• Choose a HARMONIOUS palette: analogous, complementary, or split-complementary.
• Avoid pure primary colors (#FF0000, #00FF00, #0000FF).
  Instead prefer rich tones like #C2185B, #1565C0, #2E7D32, #F9A825.
• Mix warm and cool tones for depth and visual rhythm.
• Include at least one light/cream accent (#F5F0E1, #FFF8E7) and one
  deep/dark anchor (#2C1810, #1A1A2E).

── DESIGN PRINCIPLES ──
• ALWAYS generate MANY pieces — a quilt with only 4-6 sections looks
  unfinished and unrealistic. Real quilts have dozens of pieces.
• Vary block sizes: mix large feature pieces with small accent pieces.
• Use asymmetric balance — avoid making both halves identical.
• Every swatch should be used by at least 2 sections.
• Give the quilt a creative, evocative name that reflects the chosen style
  and color story (e.g., "Autumn Ember Star", "Ocean Breeze Cabin").
"""


# ---------------------------------------------------------------
# Style-specific strategy instructions
# ---------------------------------------------------------------
_STYLE_STRATEGIES = {
    "classic_patchwork": """\
STRATEGY — CLASSIC PATCHWORK:
Create a richly varied patchwork quilt that mixes rectangles and triangles.

CONSTRUCTION ALGORITHM:
1. Divide the canvas into an IRREGULAR grid — vary column widths (between 15-30% of width)
   and row heights (between 20-35% of height). Aim for a 3-5 column × 3-5 row layout.
2. For roughly 30-50% of grid cells, split them diagonally into two half-square triangles.
   Diagonal can go top-left→bottom-right OR top-right→bottom-left (alternate randomly).
3. Assign colors so that NO two adjacent sections share the same swatch.

COLOR PALETTE GUIDANCE:
Use warm earth tones mixed with one or two jewel-tone accents:
  e.g., #8B4513 (saddle brown), #D4A76A (tan), #C2185B (berry),
        #F5F0E1 (cream), #2E4057 (navy), #E8B960 (gold)

EXAMPLE half-square-triangle split for a cell at (0,0) to (25,30):
  Upper-left triangle:  [[0,0],[25,0],[0,30],[0,0]]
  Lower-right triangle: [[25,0],[25,30],[0,30],[25,0]]
""",

    "star_burst": """\
STRATEGY — STAR BURST:
Create a dramatic star pattern radiating from the center of the canvas.

CONSTRUCTION ALGORITHM:
1. Find the canvas center: cx = width/2, cy = height/2.
2. Place a central diamond (rotated square) at the center, roughly 20-25% of canvas size.
3. Around the diamond, create 8 triangular "star points" radiating outward — each triangle
   shares one edge with the diamond and points toward a corner or edge midpoint.
4. Fill the remaining corner and edge areas with rectangles and smaller triangles.
5. The star shape (diamond + 8 points) should cover ~50-60% of the canvas.

COLOR PALETTE GUIDANCE:
Use a radiant warm-to-cool gradient — warm/bright colors at center, cooler at edges:
  Center: #F9A825 (amber), #FF7043 (coral)
  Mid-ring: #C2185B (berry), #AB47BC (purple)
  Outer: #1565C0 (blue), #263238 (charcoal), #F5F0E1 (cream)

EXAMPLE diamond at center of a 90×108 canvas (center 45,54):
  Diamond: [[45,34],[65,54],[45,74],[25,54],[45,34]]
EXAMPLE star-point triangle (top):
  [[45,34],[65,54],[65,34],[45,34]]
""",

    "log_cabin": """\
STRATEGY — LOG CABIN:
Create a log cabin quilt with concentric rectangular strips spiraling outward.

CONSTRUCTION ALGORITHM:
1. Start with a small center rectangle (about 10-15% of canvas in each dimension).
   Place it slightly off-center for visual interest.
2. Add rectangular strips spiraling outward: right → top → left → bottom → right (wider)…
3. Each strip shares one long edge with the previous layer's boundary.
4. Create at least 5-6 concentric layers.
5. Alternate between "light side" (right + top) and "dark side" (left + bottom) colors
   to achieve the classic light-and-shadow diagonal effect.

COLOR PALETTE GUIDANCE:
Light side: #F5F0E1 (cream), #E8D5B7 (linen), #FADBD8 (blush)
Dark side:  #5B2C6F (plum), #1A5276 (teal), #7B241C (burgundy)
Center:     #F9A825 (gold) — the "hearth" of the cabin

EXAMPLE for a 90×108 canvas:
  Center:       [[35,44],[55,44],[55,64],[35,64],[35,44]]
  Strip right:  [[55,44],[65,44],[65,64],[55,64],[55,44]]
  Strip top:    [[35,34],[65,34],[65,44],[35,44],[35,34]]
  Strip left:   [[25,34],[35,34],[35,64],[25,64],[25,34]]
  Strip bottom: [[25,64],[65,64],[65,74],[25,74],[25,64]]
  …continue spiraling outward until canvas is filled.
""",


    "snowball": """\
STRATEGY — SNOWBALL / OCTAGON:
Create a snowball quilt with interlocking octagons and connector squares.

CONSTRUCTION ALGORITHM:
1. Determine a cell size that divides the canvas evenly (e.g., 18-25 units).
2. Each cell contains one octagon + 4 corner triangles.
3. The octagon has corners cut at ~25% of cell size.
4. Corner triangles at intersections form small squares/diamonds connecting the octagons.
5. Alternate between 2-3 octagon colors; use one accent for all corner pieces.

COLOR PALETTE GUIDANCE:
Soft, elegant palette:
  Octagons: #D4A5A5 (dusty rose), #A8D5BA (sage), #B8C5D6 (periwinkle)
  Corners:  #FFF8E1 (cream) or #F5E6CC (ivory)
  Accent:   #5B2C6F (deep plum)

EXAMPLE octagon in a 20×20 cell at (0,0), corner cut = 5:
  Octagon: [[5,0],[15,0],[20,5],[20,15],[15,20],[5,20],[0,15],[0,5],[5,0]]
  Top-left corner:     [[0,0],[5,0],[0,5],[0,0]]
  Top-right corner:    [[15,0],[20,0],[20,5],[15,0]]
  Bottom-right corner: [[20,15],[20,20],[15,20],[20,15]]
  Bottom-left corner:  [[0,15],[5,20],[0,20],[0,15]]
""",

    "rail_fence": """\
STRATEGY — RAIL FENCE / STRIP BLOCKS:
Create a rail fence quilt with strip blocks in alternating orientations.

CONSTRUCTION ALGORITHM:
1. Divide the canvas into a grid of equal square (or near-square) blocks.
2. Each block contains 3-4 thin rectangular strips side by side.
3. Alternate orientation: odd blocks (col+row is even) have HORIZONTAL strips,
   even blocks (col+row is odd) have VERTICAL strips.
4. This creates a woven basket-weave visual effect.
5. Randomize which colors appear in each block's strips.

COLOR PALETTE GUIDANCE:
Classic Americana or cozy cabin palette:
  #8B4513 (saddle brown), #A0522D (sienna), #DEB887 (burlywood),
  #F5F0E1 (cream), #CD853F (peru), #2C1810 (espresso), #C2185B (accent berry)

EXAMPLE horizontal-strip block at (0,0) to (20,20) with 4 strips:
  Strip 1: [[0,0],[20,0],[20,5],[0,5],[0,0]]
  Strip 2: [[0,5],[20,5],[20,10],[0,10],[0,5]]
  Strip 3: [[0,10],[20,10],[20,15],[0,15],[0,10]]
  Strip 4: [[0,15],[20,15],[20,20],[0,20],[0,15]]

EXAMPLE vertical-strip block at (20,0) to (40,20) with 4 strips:
  Strip 1: [[20,0],[25,0],[25,20],[20,20],[20,0]]
  Strip 2: [[25,0],[30,0],[30,20],[25,20],[25,0]]
  Strip 3: [[30,0],[35,0],[35,20],[30,20],[30,0]]
  Strip 4: [[35,0],[40,0],[40,20],[35,20],[35,0]]
""",

    "herringbone": """\
STRATEGY — HERRINGBONE / CHEVRON:
Create a herringbone pattern with parallelogram-like shapes forming V-shapes.

CONSTRUCTION ALGORITHM:
1. Divide the canvas into vertical columns, each ~10-15% of canvas width.
2. Within each column, create pairs of right-triangles or parallelograms
   that form V or chevron shapes.
3. Adjacent columns mirror each other to create the herringbone zigzag.
4. Fill top/bottom edges with triangles to keep the rectangular canvas boundary.

COLOR PALETTE GUIDANCE:
Bold, modern palette with high contrast:
  #E91E63 (hot pink), #3F51B5 (indigo), #009688 (teal),
  #FF9800 (orange), #607D8B (blue-grey), #FFF8E1 (cream),
  #1A1A2E (midnight), #4CAF50 (green)

EXAMPLE chevron pair in column (x=0 to x=15, y=0 to y=30):
  Piece A: [[0,0],[15,15],[15,30],[0,15],[0,0]]
  Piece B: [[0,0],[15,0],[15,15],[0,0]]
  Mirror in next column (x=15 to x=30):
  Piece C: [[15,15],[30,0],[30,15],[15,30],[15,15]]
  Piece D: [[15,0],[30,0],[15,15],[15,0]]  (triangle fill)
""",

    "checkerboard": """\
STRATEGY — CHECKERBOARD:
Create a classic checkerboard quilt with alternating colored squares.

CONSTRUCTION ALGORITHM:
1. Choose a square size that divides both width and height as evenly as possible.
   Ideal: 8-12 squares per row. Calculate: square_size = width / num_cols (round to integer).
2. Create a grid of squares. Every square is type "rectangle".
3. Assign color: if (row + col) is even → color A, else → color B.
4. If the canvas doesn't divide evenly, make the last column/row slightly wider/taller.

COLOR PALETTE GUIDANCE:
Use two tones of the SAME color family for elegance:
  Option A: Deep purple #6A5ACD + Lavender #D8BFD8
  Option B: Forest green #2E7D32 + Mint #C8E6C9
  Option C: Navy #1A237E + Sky blue #BBDEFB
Optionally add a thin border strip (1-2 units) around the edge in #F5F0E1 (cream).

EXAMPLE for 90×108 canvas with 9-unit squares (10 cols × 12 rows = 120 sections):
  Square (0,0): [[0,0],[9,0],[9,9],[0,9],[0,0]] → color A
  Square (0,1): [[9,0],[18,0],[18,9],[9,9],[9,0]] → color B
  Square (1,0): [[0,9],[9,9],[9,18],[0,18],[0,9]] → color B
  …alternate using (row + col) % 2
""",
}


def build_system_prompt() -> str:
    """Return the system-level prompt (role + rules + format + schema constraints).

    Appends an auto-generated constraint summary derived from the
    authoritative Pydantic model so the prompt "contract" stays in sync.
    """
    from app.core.models import generate_schema_summary
    return _SYSTEM_PROMPT + "\n\n" + generate_schema_summary()


def build_prompt(width: int, height: int, style: str = "classic_patchwork") -> str:
    """Build the user-level prompt for a specific quilt request."""
    strategy = _STYLE_STRATEGIES.get(style, _STYLE_STRATEGIES["classic_patchwork"])

    area = width * height
    # Scale section counts with canvas area
    min_sections = max(8, int(area / 800))
    max_sections = min(80, max(20, int(area / 300)))

    prompt = f"""Design a quilt for a {width} × {height} canvas.
You MUST use between {min_sections} and {max_sections} sections. Aim for the HIGHER end.
A design with fewer than {min_sections} sections is UNACCEPTABLE.

{strategy}

REMINDER: The canvas is exactly {width} wide and {height} tall.
All section polygons must tile it perfectly with zero gaps and zero overlaps.
Total area of all sections must equal {width * height}.
Return ONLY the JSON object."""

    return prompt
