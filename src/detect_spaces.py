"""GPT-4o vision: detect spaces + door/window clearances on a floor-plan image."""

from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from typing import Any, NotRequired, TypedDict

from openai import OpenAI
from PIL import Image

from src.catalog import list_spaces, load_catalog, normalize_space_name
from src.costing import CostTracker


class DetectedSpace(TypedDict):
    plan_label: str
    catalog_space: str
    doors: NotRequired[list[str]]
    windows: NotRequired[list[str]]
    clearances: NotRequired[list[str]]


def _png_b64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def detect_spaces(
    image: Image.Image,
    *,
    client: OpenAI | None = None,
    model: str = "gpt-4o",
    cost_tracker: CostTracker | None = None,
    catalog: dict[str, Any] | None = None,
) -> list[DetectedSpace]:
    """Return plan labels, catalog mapping, and per-room clearance notes."""
    catalog = catalog or load_catalog()
    known = list_spaces(catalog)
    client = client or OpenAI()

    system = (
        "You analyze architectural floor-plan images for any residential layout. "
        "Read space labels exactly as printed. Note doors (with swing), windows, "
        "and keep-clear zones per space. Respond with JSON only."
    )
    user_text = (
        "Find every labeled SPACE/ROOM/AREA on this floor plan "
        "(ignore furniture/object labels).\n"
        "For each space return:\n"
        '- "plan_label": EXACT space name text on the plan '
        "(preserve spelling/punctuation/abbreviations).\n"
        '- "catalog_space": closest name from this catalog: '
        f"{json.dumps(known)}\n"
        '- "doors": list of short notes — wall/side of each door into this space '
        "and swing direction into the room (from the drawn swing arc).\n"
        '- "windows": list of short notes — wall/side of each window in this space.\n'
        '- "clearances": list of short keep-clear notes for furniture placement '
        "(door swing arcs, circulation from doors, avoid blocking openings).\n"
        "Omit spaces with no catalog match.\n"
        "Return JSON:\n"
        '{"spaces": [{"plan_label": "...", "catalog_space": "...", '
        '"doors": ["..."], "windows": ["..."], "clearances": ["..."]}, ...]}'
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
            label="Detect spaces + clearances",
            model=model,
            input_tokens=usage.prompt_tokens or 0,
            output_tokens=usage.completion_tokens or 0,
        )

    content = response.choices[0].message.content or "{}"
    return _parse_detected_spaces(content, catalog)


def constraints_by_label(
    detected: list[DetectedSpace],
) -> dict[str, dict[str, list[str]]]:
    """Map plan_label → doors/windows/clearances for furnish prompts."""
    out: dict[str, dict[str, list[str]]] = {}
    for item in detected:
        label = item.get("plan_label")
        if not label:
            continue
        out[label] = {
            "doors": list(item.get("doors") or []),
            "windows": list(item.get("windows") or []),
            "clearances": list(item.get("clearances") or []),
        }
    return out


def _parse_detected_spaces(
    content: str,
    catalog: dict[str, Any],
) -> list[DetectedSpace]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return []
        data = json.loads(match.group(0))

    raw = data.get("spaces") or data.get("rooms") or []
    if isinstance(raw, str):
        raw = [raw]

    known = set(list_spaces(catalog))
    results: list[DetectedSpace] = []
    seen_labels: set[str] = set()

    for item in raw:
        plan_label = ""
        catalog_hint = ""
        doors: list[str] = []
        windows: list[str] = []
        clearances: list[str] = []

        if isinstance(item, dict):
            plan_label = str(
                item.get("plan_label") or item.get("label") or item.get("name") or ""
            ).strip()
            catalog_hint = str(
                item.get("catalog_space") or item.get("catalog") or ""
            ).strip()
            doors = _as_str_list(item.get("doors"))
            windows = _as_str_list(item.get("windows"))
            clearances = _as_str_list(item.get("clearances"))
        else:
            plan_label = str(item).strip()

        if not plan_label or plan_label in seen_labels:
            continue

        catalog_space = (
            normalize_space_name(catalog_hint, catalog) if catalog_hint else None
        )
        if catalog_space is None:
            catalog_space = normalize_space_name(plan_label, catalog)
        if catalog_space is None or catalog_space not in known:
            continue

        seen_labels.add(plan_label)
        entry: DetectedSpace = {
            "plan_label": plan_label,
            "catalog_space": catalog_space,
        }
        if doors:
            entry["doors"] = doors
        if windows:
            entry["windows"] = windows
        if clearances:
            entry["clearances"] = clearances
        results.append(entry)

    return results
