"""gpt-image-2 clean + furnish helpers."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from openai import OpenAI
from PIL import Image

from src.costing import CostTracker

IMAGE_MODEL = "gpt-image-2"


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
) -> Image.Image:
    file_obj = _to_png_file(image)
    kwargs: dict[str, Any] = {
        "model": model,
        "image": file_obj,
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
        "input_fidelity": "high",
    }
    try:
        response = client.images.edit(**kwargs)
    except Exception as first_err:
        # Retry with a minimal parameter set for broader model compatibility
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
    """Remove furniture, dimensions, labels, and measurements; keep wall structure."""
    client = client or OpenAI()
    prompt = (
        "Edit this architectural floor plan into a CLEAN 2D top-down line drawing.\n"
        "REMOVE all furniture, fixtures that are movable, appliances drawn as furniture, "
        "dimension strings, measurement numbers, room name labels, text annotations, "
        "arrows, and callouts.\n"
        "KEEP walls, doors, windows, openings, and the overall room geometry accurately.\n"
        "Output must be a visually clean 2D floor plan with empty rooms, no text, "
        "no dimensions, no labels, no measurements. White/light background, dark walls."
    )
    result = _edit_image(client, image, prompt, model=model)
    if cost_tracker is not None:
        cost_tracker.add_image(
            label="Clean floor plan (wipe furniture + labels)",
            model=model,
            images=1,
            details="gpt-image-2 edit — clean empty plan",
        )
    return result


def furnish_floor_plan(
    clean_image: Image.Image,
    selections: dict[str, list[str]],
    *,
    client: OpenAI | None = None,
    model: str = IMAGE_MODEL,
    cost_tracker: CostTracker | None = None,
) -> Image.Image:
    """Place selected furniture on the clean plan in one image edit."""
    client = client or OpenAI()
    lines = []
    for space, items in selections.items():
        if not items:
            continue
        lines.append(f"- {space}: {', '.join(items)}")
    selection_block = "\n".join(lines) if lines else "- (none)"

    prompt = (
        "You are given a CLEAN empty 2D architectural floor plan (top-down).\n"
        "Place ONLY the following furniture into the matching rooms:\n"
        f"{selection_block}\n\n"
        "Placement rules (strict):\n"
        "- Keep the exact same wall layout, doors, and windows.\n"
        "- Use clear 2D top-down architectural furniture symbols (plan view), visually polished.\n"
        "- Scale furniture realistically to each room.\n"
        "- No overlapping furniture; leave circulation paths; do not block doors or windows.\n"
        "- Align beds/sofas along walls sensibly; beds with head against a wall; wardrobes on walls.\n"
        "- No awkward floating pieces in the middle of circulation.\n"
        "- Do NOT add any text, room labels, dimensions, measurements, or annotations.\n"
        "- Do NOT add furniture that was not listed.\n"
        "- Output one complete furnished 2D floor plan image."
    )
    result = _edit_image(client, clean_image, prompt, model=model)
    if cost_tracker is not None:
        cost_tracker.add_image(
            label="Furnish floor plan (all selections)",
            model=model,
            images=1,
            details="gpt-image-2 edit — place selected furniture",
        )
    return result
