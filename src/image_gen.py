"""gpt-image-2 clean + furnish helpers."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from openai import OpenAI
from PIL import Image

from src.costing import CostTracker

IMAGE_MODEL = "gpt-image-2"

# Per-room constraints from detection (doors / windows / clearances)
RoomConstraints = dict[str, Any]


def _to_png_file(image: Image.Image, max_side: int = 1536) -> BytesIO:
    img = image.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    buf.name = "floorplan.png"
    return buf


def _decode_image_response(response: Any) -> Image.Image:
    item = response.data[0]
    b64 = getattr(item, "b64_json", None)
    if b64:
        raw = base64.b64decode(b64)
        return Image.open(BytesIO(raw)).convert("RGB")
    url = getattr(item, "url", None)
    if url:
        import urllib.request

        with urllib.request.urlopen(url) as resp:
            raw = resp.read()
        return Image.open(BytesIO(raw)).convert("RGB")
    raise RuntimeError("Image API returned neither b64_json nor url.")


def _edit_image(
    client: OpenAI,
    image: Image.Image,
    prompt: str,
    *,
    model: str = IMAGE_MODEL,
    input_fidelity: str | None = "high",
) -> Image.Image:
    file_obj = _to_png_file(image)
    kwargs: dict[str, Any] = {
        "model": model,
        "image": file_obj,
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
    }
    if input_fidelity:
        kwargs["input_fidelity"] = input_fidelity
    try:
        response = client.images.edit(**kwargs)
    except Exception as first_err:
        file_obj.seek(0)
        try:
            response = client.images.edit(
                model=model,
                image=file_obj,
                prompt=prompt,
                n=1,
            )
        except Exception as second_err:
            raise RuntimeError(
                f"{model} image edit failed. "
                f"First error: {first_err}. Retry error: {second_err}"
            ) from second_err
    return _decode_image_response(response)


def clean_floor_plan(
    image: Image.Image,
    *,
    client: OpenAI | None = None,
    model: str = IMAGE_MODEL,
    cost_tracker: CostTracker | None = None,
) -> Image.Image:
    """Strip furniture/object labels; keep structure, door swings, and space names."""
    client = client or OpenAI()
    prompt = (
        "Edit this architectural floor plan into a CLEAN 2D top-down line drawing. "
        "Rules are GLOBAL and must work for any residential floor-plan PDF.\n\n"
        "NO INVENTION (never violate):\n"
        "- Do not add walls, rooms, doors, windows, openings, fixtures, furniture, or "
        "labels that are not already on the uploaded plan.\n"
        "- Do not complete, symmetrize, straighten, or restyle the architecture.\n"
        "- Do not invent new geometry. If it is not on the source, it must not appear.\n\n"
        "KEEP (structure):\n"
        "- Walls, columns, windows, openings — exact geometry; do not erase, move, "
        "thin, open, break, or merge walls.\n"
        "- Door leaves AND door swing arcs exactly as drawn.\n\n"
        "KEEP (text) — space/room/area names ONLY:\n"
        "- Retain a text label only if it names a SPACE (a room, toilet/bath, kitchen, "
        "living/dining area, foyer/lobby/entrance, balcony/terrace, utility/laundry, "
        "study/office, wardrobe room, store, or similar area).\n"
        "- Preserve the exact spelling of those space names as on the plan.\n"
        "- If a label mixes measurements with a space name, redraw it as the space "
        "name only (drop numbers, units, and width/length prefixes).\n\n"
        "REMOVE completely:\n"
        "- All furniture and fixture drawings/symbols (beds, sofas, tables, chairs, "
        "wardrobes, counters drawn as furniture, appliances, rugs, etc.).\n"
        "- All OBJECT/FIXTURE text labels — any label that names furniture, storage, "
        "seating, appliances, equipment, or fittings rather than a room/space "
        "(examples of the CATEGORY to remove, not a fixed list: wardrobe, dresser, "
        "side table, TV unit, storage, seating, study table, appliance, refrigerator, "
        "washing machine, mandir unit, etc.).\n"
        "- Dimension strings, measurement numbers, arrows, and non-space callouts.\n\n"
        "Output: empty rooms with walls/doors/swings/windows + space names only. "
        "No furniture geometry. No object labels. No dimensions. "
        "Nothing that was not on the uploaded plan. White/light background, dark walls."
    )
    result = _edit_image(client, image, prompt, model=model, input_fidelity="high")
    if cost_tracker is not None:
        cost_tracker.add_image(
            label="Clean floor plan (space names + door swings; no furniture)",
            model=model,
            images=1,
            details="gpt-image-2 edit — clean plan with space labels",
        )
    return result


def _format_constraints_block(
    selections: dict[str, list[str]],
    constraints: dict[str, RoomConstraints] | None,
) -> str:
    if not constraints:
        return (
            "No per-room clearance notes were provided. Still obey universal "
            "door-swing and window rules below."
        )
    parts: list[str] = []
    for room in selections:
        if not selections.get(room):
            continue
        info = constraints.get(room) or {}
        doors = info.get("doors") or []
        windows = info.get("windows") or []
        clearances = info.get("clearances") or []
        lines = [f'Room "{room}":']
        if doors:
            lines.append("  Doors/swings: " + "; ".join(str(d) for d in doors))
        if windows:
            lines.append("  Windows: " + "; ".join(str(w) for w in windows))
        if clearances:
            lines.append("  Clearances: " + "; ".join(str(c) for c in clearances))
        if len(lines) == 1:
            lines.append("  (no extra notes — still clear all door swings)")
        parts.append("\n".join(lines))
    return "\n".join(parts) if parts else (
        "No matching clearance notes for selected rooms. "
        "Still obey universal door-swing and window rules below."
    )


def furnish_floor_plan(
    clean_image: Image.Image,
    selections: dict[str, list[str]],
    *,
    client: OpenAI | None = None,
    model: str = IMAGE_MODEL,
    cost_tracker: CostTracker | None = None,
    constraints: dict[str, RoomConstraints] | None = None,
) -> Image.Image:
    """Place selected furniture using plan labels + optional per-room clearances.

    ``selections`` keys must be the exact plan room labels visible on the image.
    ``constraints`` is keyed by the same plan labels (from detection).
    """
    client = client or OpenAI()
    lines = []
    for space, items in selections.items():
        if not items:
            continue
        lines.append(f'- Room labeled "{space}": {", ".join(items)}')
    selection_block = "\n".join(lines) if lines else "- (none)"
    constraints_block = _format_constraints_block(selections, constraints)

    prompt = (
        "You are given a CLEAN 2D architectural floor plan (top-down). "
        "Space name labels and door swing arcs are on the plan. "
        "These rules are GLOBAL for any floor plan.\n\n"
        "Place ONLY the following furniture, each item ONLY inside the room whose "
        "on-plan label matches the name in quotes:\n"
        f"{selection_block}\n\n"
        "PER-ROOM OPENINGS / CLEARANCES (from plan analysis):\n"
        f"{constraints_block}\n\n"
        "STRUCTURE LOCK (never violate):\n"
        "- Copy walls, columns, doors, door swing arcs, windows, and openings EXACTLY.\n"
        "- Do not erase, move, thin, open, break, or merge walls/rooms.\n"
        "- Keep every existing space name label; do not add dimensions or measurements.\n\n"
        "PLACEMENT RULES (strict, all rooms):\n"
        "- Place each item only in its matching labeled room; never cross into another space.\n"
        "- NEVER place furniture inside a door swing arc, against a door leaf, or so it "
        "blocks the clear path from any door into the room.\n"
        "- Prefer walls that are free of doors for large pieces (beds, wardrobes, sofas, "
        "storage). If a wall has a door, leave the swing and approach clear.\n"
        "- TALL furniture (wardrobes, TV units, storage, beds, bookcases, dressers) must "
        "NOT sit against or cover windows. Use a free wall instead.\n"
        "- SMALL / LOW furniture (side tables, stools, chairs) MAY sit against a window "
        "wall if they do not block the window opening.\n"
        "- Use correct 2D plan symbols and true shapes (e.g. L-shaped sofa must look L-shaped).\n"
        "- Scale each piece to the room: not oversized, not tiny; leave clear circulation; "
        "no overlapping pieces.\n"
        "- Beds: head against a wall that is NOT in a door swing path and NOT a window wall.\n"
        "- Sofas/TV units/wardrobes/storage: against walls, not floating in circulation.\n"
        "- Do NOT add furniture that was not listed.\n\n"
        "FURNITURE LOOK (furniture only — do not restyle the plan):\n"
        "- Render furniture as clean modern 2D plan symbols with filled color, not bare outlines only.\n"
        "- Palette: beige / cream / soft grey for upholstery and bedding; light wood / pale tan "
        "accents on case-good tops (tables, dressers, side tables, desks).\n"
        "- Soft subtle quilt or square-cushion texture on seats (sofas, chairs, armchairs) only.\n"
        "- Wardrobes/storage: flat modern fills (white or light grey) with thin outlines.\n"
        "- Do NOT use dark wood, heavy grain, bold fabric prints, or busy patterns.\n"
        "- Walls, columns, doors, door swing arcs, windows, openings, and space labels must stay "
        "as black/white line drawing.\n"
        "- Output one complete furnished 2D floor plan image."
    )
    result = _edit_image(
        client, clean_image, prompt, model=model, input_fidelity="high"
    )
    if cost_tracker is not None:
        cost_tracker.add_image(
            label="Furnish floor plan (all selections)",
            model=model,
            images=1,
            details="gpt-image-2 edit — place selected furniture",
        )
    return result


def refine_furnished_plan(
    furnished_image: Image.Image,
    user_prompt: str,
    *,
    client: OpenAI | None = None,
    model: str = IMAGE_MODEL,
    cost_tracker: CostTracker | None = None,
) -> Image.Image:
    """Apply a user prompt to an already furnished 2D floor plan."""
    client = client or OpenAI()
    instruction = user_prompt.strip()
    prompt = (
        "You are given an already FURNISHED 2D architectural floor plan (top-down).\n\n"
        "PRIMARY TASK — the output is WRONG if this is not visibly done:\n"
        f'"{instruction}"\n\n'
        "You MUST follow that instruction strictly.\n"
        "- If the user asks to swap or move furniture, relocate those pieces. Their old "
        "positions must be empty of those pieces. Do not leave them where they started.\n"
        "- If the user asks for a piece to be stuck / flush / against a wall, draw it "
        "tight to that wall (not floating).\n"
        "- A copy of the input with the same furniture layout is a failed result.\n\n"
        "STRUCTURE LOCK (walls only — never violate):\n"
        "- Copy walls, columns, doors, door swing arcs, windows, and openings EXACTLY.\n"
        "- Do not erase, move, thin, open, break, or merge walls/rooms.\n"
        "- Keep every existing space name label; do not add dimensions or measurements.\n"
        "- Walls, columns, doors, door swing arcs, windows, openings, and space labels "
        "must stay as black/white line drawing.\n"
        "- Leave other rooms and unmentioned furniture unchanged.\n\n"
        "AFTER MOVING, still obey openings:\n"
        "- Never place furniture inside a door swing arc or blocking a door path.\n"
        "- Tall furniture (wardrobes, TV units, storage, beds, bookcases) must not sit "
        "against or cover windows.\n"
        "- Small/low furniture may sit against a window wall if it does not block the opening.\n"
        "- Output one complete furnished 2D floor plan image with the requested layout change."
    )
    result = _edit_image(
        client, furnished_image, prompt, model=model, input_fidelity="low"
    )
    if cost_tracker is not None:
        cost_tracker.add_image(
            label="Refine furnished plan from user prompt",
            model=model,
            images=1,
            details="gpt-image-2 edit — user prompt on furnished plan",
        )
    return result
