"""Streamlit POC: floor-plan PDF → clean plan → furnished plan."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from src.catalog import furniture_for_space, list_spaces, load_catalog
from src.costing import USD_TO_INR, CostTracker
from src.detect_spaces import detect_spaces
from src.image_gen import IMAGE_MODEL, clean_floor_plan, furnish_floor_plan
from src.pdf_utils import image_to_png_bytes, pdf_bytes_to_png

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

st.set_page_config(
    page_title="Personalised Floor Plan Furniture",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_FURNITURE = 3


def _init_state() -> None:
    defaults = {
        "original_image": None,
        "clean_image": None,
        "final_image": None,
        "detected_spaces": None,
        "cost_rows": [],
        "show_cost": True,
        "upload_token": None,
        "selections": {},
        "skip_detection": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_run_state() -> None:
    st.session_state.original_image = None
    st.session_state.clean_image = None
    st.session_state.final_image = None
    st.session_state.detected_spaces = None
    st.session_state.cost_rows = []
    st.session_state.selections = {}
    st.session_state.skip_detection = True


def _tracker() -> CostTracker:
    return CostTracker.from_dicts(st.session_state.cost_rows)


def _persist_tracker(tracker: CostTracker) -> None:
    st.session_state.cost_rows = tracker.to_dicts()


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key.startswith("sk-your-key"):
        raise RuntimeError(
            "OPENAI_API_KEY missing. Copy .env.example to .env and set your key."
        )
    return OpenAI(api_key=api_key)


def _render_cost_panel() -> None:
    st.sidebar.markdown("### API cost (estimated)")
    toggle_label = "Hide cost" if st.session_state.show_cost else "Show cost"
    if st.sidebar.button(toggle_label, key="toggle_cost"):
        st.session_state.show_cost = not st.session_state.show_cost
        st.rerun()

    if not st.session_state.show_cost:
        st.sidebar.caption("Cost panel hidden.")
        return

    tracker = _tracker()
    if not tracker.entries:
        st.sidebar.caption("No API calls in this run yet.")
        return

    for entry in tracker.entries:
        st.sidebar.markdown(
            f"**{entry.label}**  \n"
            f"`{entry.model}` · {entry.call_type}  \n"
            f"${entry.estimated_usd:.4f} · ₹{entry.estimated_inr:.2f}  \n"
            f"_{entry.details}_"
        )
        st.sidebar.divider()

    st.sidebar.markdown(
        f"### Run total  \n"
        f"**${tracker.total_usd:.4f}** · **₹{tracker.total_inr:.2f}**"
    )
    st.sidebar.caption(
        f"Estimates only. Vision uses GPT-4o token rates; images use "
        f"assumed {IMAGE_MODEL} unit price. INR at $1 = ₹{USD_TO_INR:.0f}."
    )


def main() -> None:
    _init_state()
    catalog = load_catalog()
    all_spaces = list_spaces(catalog)

    st.title("Personalised Floor Plan Furniture")
    st.caption(
        "Upload a floor-plan PDF → detect spaces → pick furniture (max 3 per space) "
        "→ generate one clean 2D furnished plan (no dimensions/labels)."
    )

    _render_cost_panel()

    uploaded = st.file_uploader("Upload floor plan (PDF only)", type=["pdf"])
    if uploaded is None:
        st.info("Upload a PDF from `floor-plans/` to begin.")
        return

    token = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.upload_token != token:
        _reset_run_state()
        st.session_state.upload_token = token
        try:
            st.session_state.original_image = pdf_bytes_to_png(
                uploaded.getvalue(), dpi=200
            )
        except Exception as exc:
            st.error(f"Failed to read PDF: {exc}")
            return

    original: Image.Image | None = st.session_state.original_image
    if original is None:
        st.error("Could not rasterize PDF.")
        return

    st.subheader("1. Uploaded plan")
    st.image(original, use_container_width=True, caption=uploaded.name)

    st.subheader("2. Detect spaces")
    col_d1, col_d2 = st.columns([1, 3])
    with col_d1:
        detect_clicked = st.button("Detect spaces (GPT-4o)", type="primary")
    with col_d2:
        st.session_state.skip_detection = st.checkbox(
            "Skip detection — show all catalog spaces (offline UI check)",
            value=st.session_state.skip_detection,
        )

    if detect_clicked:
        try:
            with st.spinner("Detecting spaces with GPT-4o…"):
                client = _get_client()
                tracker = _tracker()
                spaces = detect_spaces(
                    original,
                    client=client,
                    cost_tracker=tracker,
                    catalog=catalog,
                )
                _persist_tracker(tracker)
                st.session_state.detected_spaces = spaces or []
                st.session_state.skip_detection = False
                if not spaces:
                    st.warning(
                        "No catalog spaces matched. Try another PDF or use all spaces."
                    )
                else:
                    st.success(f"Detected: {', '.join(spaces)}")
        except Exception as exc:
            st.exception(exc)

    if st.session_state.skip_detection or st.session_state.detected_spaces is None:
        available_spaces = all_spaces
        if st.session_state.detected_spaces is None and not st.session_state.skip_detection:
            st.caption("Run detection, or keep skip enabled to use the full catalog.")
    else:
        available_spaces = [
            s for s in st.session_state.detected_spaces if s in all_spaces
        ] or all_spaces

    st.subheader("3. Configure furniture (multiple spaces)")
    detected = st.session_state.detected_spaces or []
    default_spaces = [s for s in detected if s in available_spaces][:4]

    selected_spaces = st.multiselect(
        "Spaces to furnish",
        options=available_spaces,
        default=default_spaces,
        help="Only spaces available on the plan / catalog are listed.",
    )

    selections: dict[str, list[str]] = {}
    for space in selected_spaces:
        options = furniture_for_space(space, catalog)
        prior = [
            c for c in st.session_state.selections.get(space, []) if c in options
        ][:MAX_FURNITURE]
        picked = st.multiselect(
            f"Furniture for **{space}** (max {MAX_FURNITURE})",
            options=options,
            default=prior,
            key=f"furn_{space}",
            max_selections=MAX_FURNITURE,
        )
        selections[space] = list(picked)[:MAX_FURNITURE]

    st.session_state.selections = selections

    ready = any(bool(items) for items in selections.values())
    st.subheader("4. Generate")
    gen = st.button(
        "Generate clean + furnished plan",
        type="primary",
        disabled=not ready,
        help=(
            "Runs gpt-image-2: wipe furniture/labels, then place all selections "
            "in one final image."
        ),
    )

    if gen and ready:
        try:
            client = _get_client()
            tracker = _tracker()
            with st.spinner("Step A — cleaning floor plan (gpt-image-2)…"):
                clean = clean_floor_plan(original, client=client, cost_tracker=tracker)
                st.session_state.clean_image = clean
            with st.spinner("Step B — placing furniture (gpt-image-2)…"):
                final = furnish_floor_plan(
                    clean,
                    {k: v for k, v in selections.items() if v},
                    client=client,
                    cost_tracker=tracker,
                )
                st.session_state.final_image = final
            _persist_tracker(tracker)
            st.success("Generation complete.")
        except Exception as exc:
            st.exception(exc)

    clean_img = st.session_state.clean_image
    final_img = st.session_state.final_image
    if clean_img is not None or final_img is not None:
        st.subheader("5. Results")
        cols = st.columns(3)
        with cols[0]:
            st.markdown("**Original**")
            st.image(original, use_container_width=True)
        with cols[1]:
            st.markdown("**Clean (no furniture / labels)**")
            if clean_img is not None:
                st.image(clean_img, use_container_width=True)
                st.download_button(
                    "Download clean PNG",
                    data=image_to_png_bytes(clean_img),
                    file_name="clean_floor_plan.png",
                    mime="image/png",
                    key="dl_clean",
                )
            else:
                st.caption("Not generated yet.")
        with cols[2]:
            st.markdown("**Final furnished**")
            if final_img is not None:
                st.image(final_img, use_container_width=True)
                st.download_button(
                    "Download final PNG",
                    data=image_to_png_bytes(final_img),
                    file_name="furnished_floor_plan.png",
                    mime="image/png",
                    key="dl_final",
                )
            else:
                st.caption("Not generated yet.")

    with st.expander("Selection summary"):
        st.json(
            {k: v for k, v in selections.items() if v}
            or {"note": "No furniture selected"}
        )


if __name__ == "__main__":
    main()
