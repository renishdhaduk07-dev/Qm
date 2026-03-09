"""
Streamlit frontend for the AI Quilt Generator.
Collects width & height, calls backend API, displays the rendered quilt image.
"""

import base64
import io

import requests
import streamlit as st

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="AI Quilt Generator",
    page_icon="🧵",
    layout="centered",
)

BACKEND_URL = "http://localhost:8000"

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.title("🧵 AI Quilt Generator")
st.markdown(
    "Enter a canvas size below and click **Generate** to create "
    "an AI-designed quilt pattern."
)

# ---------------------------------------------------------------
# Input controls
# ---------------------------------------------------------------
PRESET_SIZES = {
    '30 × 40  (Small Throw)': (30, 40),
    '70 × 90  (Twin)': (70, 90),
    '90 × 108 (Queen)': (90, 108),
    '106 × 112 (King)': (106, 112),
    "Custom": None,
}

size_choice = st.selectbox("Quilt Size", list(PRESET_SIZES.keys()))

if PRESET_SIZES[size_choice] is not None:
    width, height = PRESET_SIZES[size_choice]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Width (inches)", f'{width}"')
    with col2:
        st.metric("Height (inches)", f'{height}"')
else:
    col1, col2 = st.columns(2)
    with col1:
        width = st.number_input(
            "Width (inches)", min_value=1, value=100, step=1
        )
    with col2:
        height = st.number_input(
            "Height (inches)", min_value=1, value=100, step=1
        )

# ---------------------------------------------------------------
# Pattern style
# ---------------------------------------------------------------
PATTERN_STYLES = {
    "🧩 Classic Patchwork": "classic_patchwork",
    "⭐ Star Burst": "star_burst",
    "🏠 Log Cabin": "log_cabin",
    "🎱 Snowball / Octagon": "snowball",
    "🚧 Rail Fence": "rail_fence",
    "🪚 Herringbone / Chevron": "herringbone",
    "♟️ Checkerboard": "checkerboard",
}

style_label = st.selectbox("Pattern Style", list(PATTERN_STYLES.keys()))
style = PATTERN_STYLES[style_label]

generate_btn = st.button("🎨 Generate Quilt", use_container_width=True)

# ---------------------------------------------------------------
# Session state
# ---------------------------------------------------------------
if "quilt_data" not in st.session_state:
    st.session_state.quilt_data = None
if "quilt_settings" not in st.session_state:
    st.session_state.quilt_settings = None


def _call_backend(w, h, s):
    """Call the backend and return quilt dict or None."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/generate-quilt",
            json={"width": w, "height": h, "style": s},
            timeout=600,
        )
        if resp.status_code != 200:
            st.error(
                f"Backend error ({resp.status_code}): "
                f"{resp.json().get('detail', resp.text)}"
            )
            return None
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to backend. Make sure the server is running "
            f"at `{BACKEND_URL}`."
        )
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


# ---------------------------------------------------------------
# Handle Generate button
# ---------------------------------------------------------------
if generate_btn:
    with st.spinner("Generating quilt pattern with AI…"):
        result = _call_backend(width, height, style)
        if result:
            st.session_state.quilt_data = result
            st.session_state.quilt_settings = {
                "width": width, "height": height, "style": style,
            }
            st.rerun()

# ---------------------------------------------------------------
# Display quilt if available
# ---------------------------------------------------------------
if st.session_state.quilt_data is not None:
    quilt = st.session_state.quilt_data

    st.success(f"✅ Quilt generated: **{quilt.get('name', 'Untitled')}**")
    st.caption(
        f"Size: {quilt['size']['width']} × {quilt['size']['height']} · "
        f"{len(quilt['sections'])} sections · "
        f"{len(quilt['swatches'])} swatches"
    )

    # ---------------------------------------------------
    # Display the backend-rendered image
    # ---------------------------------------------------
    image_b64 = quilt.get("image")
    if image_b64:
        img_bytes = base64.b64decode(image_b64)
        st.image(img_bytes, caption=quilt.get("name", "Quilt"),
                 use_container_width=True)
    else:
        st.warning("No rendered image available from backend.")

    # ---------------------------------------------------
    # Action buttons: Regenerate & Save
    # ---------------------------------------------------
    col_regen, col_save = st.columns(2)
    with col_regen:
        if st.button("🔄 Regenerate", use_container_width=True):
            s = st.session_state.quilt_settings
            if s:
                with st.spinner("Generating a fresh design…"):
                    result = _call_backend(s["width"], s["height"], s["style"])
                    if result:
                        st.session_state.quilt_data = result
                        st.rerun()
    with col_save:
        import json as _json
        # Save JSON without the image field to keep file small
        save_data = {k: v for k, v in quilt.items() if k != "image"}
        st.download_button(
            "💾 Save JSON",
            data=_json.dumps(save_data, indent=2),
            file_name=f"{quilt.get('name', 'quilt').replace(' ', '_')}.json",
            mime="application/json",
            use_container_width=True,
        )

    with st.expander("📋 View raw JSON"):
        # Show JSON without bulky base64 image
        display_data = {k: v for k, v in quilt.items() if k != "image"}
        st.json(display_data)
