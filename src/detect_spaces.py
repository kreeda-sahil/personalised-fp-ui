"""GPT-4o vision: detect spaces present on a floor-plan image."""

from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from typing import Any

from openai import OpenAI
from PIL import Image

from src.catalog import list_spaces, load_catalog, map_detected_to_catalog
from src.costing import CostTracker


def _png_b64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def detect_spaces(
    image: Image.Image,
    *,
    client: OpenAI | None = None,
    model: str = "gpt-4o",
    cost_tracker: CostTracker | None = None,
    catalog: dict[str, Any] | None = None,
) -> list[str]:
    """Return catalog space names detected on the floor plan."""
    catalog = catalog or load_catalog()
    known = list_spaces(catalog)
    client = client or OpenAI()

    system = (
        "You analyze architectural floor-plan images. "
        "Identify which rooms/spaces are present from labels and layout. "
        "Respond with JSON only."
    )
    user_text = (
        "List the spaces visible on this floor plan.\n"
        f"Prefer names from this catalog when they match: {json.dumps(known)}\n"
        "If a room is clearly present but named differently, use the closest catalog name.\n"
        'Return JSON: {"spaces": ["Living Room", "Kitchen", ...]}'
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{_png_b64(image)}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    usage = response.usage
    if cost_tracker is not None and usage is not None:
        cost_tracker.add_vision(
            label="Detect spaces",
            model=model,
            input_tokens=usage.prompt_tokens or 0,
            output_tokens=usage.completion_tokens or 0,
        )

    content = response.choices[0].message.content or "{}"
    raw_spaces = _parse_spaces_json(content)
    return map_detected_to_catalog(raw_spaces, catalog)


def _parse_spaces_json(content: str) -> list[str]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return []
        data = json.loads(match.group(0))

    spaces = data.get("spaces") or data.get("rooms") or []
    if isinstance(spaces, str):
        spaces = [spaces]
    return [str(s).strip() for s in spaces if str(s).strip()]
