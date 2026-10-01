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
from src.detect_spaces import DetectedSpace, constraints_by_label, detect_spaces
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
        "detected_spaces": None,  # list[DetectedSpace] | None
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


def _as_detected_list(raw: list | None) -> list[DetectedSpace]:
    """Normalize session state (legacy string list or DetectedSpace dicts)."""
    if not raw:
        return []
    out: list[DetectedSpace] = []
    for item in raw:
        if isinstance(item, dict) and item.get("plan_label") and item.get("catalog_space"):
            entry: DetectedSpace = {
                "plan_label": str(item["plan_label"]),
                "catalog_space": str(item["catalog_space"]),
            }
            for key in ("doors", "windows", "clearances"):
                val = item.get(key)
                if isinstance(val, list) and val:
                    entry[key] = [str(v) for v in val if str(v).strip()]  # type: ignore[literal-required]
            out.append(entry)
        elif isinstance(item, str) and item.strip():
            out.append({"plan_label": item, "catalog_space": item})
    return out


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
    all_catalog_spaces = list_spaces(catalog)

    st.title("Personalised Floor Plan Furniture")
    st.caption(
        "Upload a floor-plan PDF → detect spaces + clearances → pick furniture "
        "(max 3 per space) → clean plan (space names + door swings) → furnish."
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
        detect_clicked = st.button("Detect spaces", type="primary")
    with col_d2:
        st.session_state.skip_detection = st.checkbox(
            "Skip detection — show all catalog spaces (offline UI check)",
            value=st.session_state.skip_detection,
        )

    if detect_clicked:
        try:
            with st.spinner("Detecting spaces + clearances"):
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
                    shown = ", ".join(
                        f"{s['plan_label']} → {s['catalog_space']}" for s in spaces
                    )
                    st.success(f"Detected: {shown}")
        except Exception as exc:
            st.exception(exc)

    detected = _as_detected_list(st.session_state.detected_spaces)
    room_constraints = constraints_by_label(detected)

    # plan_label → catalog_space for furniture lookup
    label_to_catalog: dict[str, str] = {}
    if st.session_state.skip_detection or not detected:
        available_labels = all_catalog_spaces
        for name in all_catalog_spaces:
            label_to_catalog[name] = name
        if not detected and not st.session_state.skip_detection:
            st.caption("Run detection, or keep skip enabled to use the full catalog.")
    else:
        available_labels = []
        for item in detected:
            label = item["plan_label"]
            cat = item["catalog_space"]
            if cat not in all_catalog_spaces:
                continue
            if label not in label_to_catalog:
                available_labels.append(label)
                label_to_catalog[label] = cat
        if not available_labels:
            available_labels = all_catalog_spaces
            for name in all_catalog_spaces:
                label_to_catalog[name] = name

    st.subheader("3. Configure furniture (multiple spaces)")
    default_labels = [lbl for lbl in available_labels if lbl in label_to_catalog][:4]
    if detected and not st.session_state.skip_detection:
        default_labels = [
            d["plan_label"] for d in detected if d["plan_label"] in available_labels
        ][:4]

    selected_labels = st.multiselect(
        "Spaces to furnish",
        options=available_labels,
        default=default_labels,
        help="Names match the floor-plan labels when detection is used.",
    )

    selections: dict[str, list[str]] = {}
    for plan_label in selected_labels:
        catalog_space = label_to_catalog.get(plan_label, plan_label)
        options = furniture_for_space(catalog_space, catalog)
        prior = [
            c for c in st.session_state.selections.get(plan_label, []) if c in options
        ][:MAX_FURNITURE]
        picked = st.multiselect(
            f"Furniture for **{plan_label}** (max {MAX_FURNITURE})",
            options=options,
            default=prior,
            key=f"furn_{plan_label}",
            max_selections=MAX_FURNITURE,
        )
        selections[plan_label] = list(picked)[:MAX_FURNITURE]

    st.session_state.selections = selections

    ready = any(bool(items) for items in selections.values())
    st.subheader("4. Generate")
    gen = st.button(
        "Generate clean + furnished plan",
        type="primary",
        disabled=not ready,
        help=(
            "Runs gpt-image-2: clean (space names + door swings), then place "
            "furniture using clearance notes from detection when available."
        ),
    )

    if gen and ready:
        try:
            client = _get_client()
            tracker = _tracker()
            active = {k: v for k, v in selections.items() if v}
            furnish_constraints = {
                k: room_constraints[k] for k in active if k in room_constraints
            } or None
            with st.spinner("Step A — cleaning floor plan"):
                clean = clean_floor_plan(original, client=client, cost_tracker=tracker)
                st.session_state.clean_image = clean
            with st.spinner("Step B — placing furniture"):
                final = furnish_floor_plan(
                    clean,
                    active,
                    client=client,
                    cost_tracker=tracker,
                    constraints=furnish_constraints,
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
            st.markdown("**Clean (space names + door swings)**")
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
        summary = {
            plan_label: {
                "catalog_space": label_to_catalog.get(plan_label, plan_label),
                "furniture": items,
                "clearances": room_constraints.get(plan_label),
            }
            for plan_label, items in selections.items()
            if items
        }
        st.json(summary or {"note": "No furniture selected"})


if __name__ == "__main__":
    main()
