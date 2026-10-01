"""Load space → furniture catalog derived from the Dezinebox quiz Excel."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Canonical spaces from the quiz picker (names only)
CANONICAL_SPACES = [
    "Living Room",
    "Dining Room",
    "Kitchen",
    "Master Bedroom",
    "Bedroom 2",
    "Bedroom 3",
    "Guest Bedroom",
    "Kids' Bedroom",
    "Study / Home Office",
    "Puja Room",
    "Entertainment / Family Room",
    "Home Theatre",
    "Bathrooms / Toilets",
    "Utility / Laundry",
    "Balcony",
    "Terrace",
    "Entrance / Foyer",
    "Walk-in Wardrobe",
    "Dressing Room",
    "Store Room",
    "Servant / Staff Room",
]

# Detection aliases → canonical catalog space
SPACE_ALIASES: dict[str, str] = {
    "living": "Living Room",
    "living room": "Living Room",
    "living/dining": "Living Room",
    "drawing room": "Living Room",
    "dining": "Dining Room",
    "dining room": "Dining Room",
    "kitchen": "Kitchen",
    "master bedroom": "Master Bedroom",
    "master bed room": "Master Bedroom",
    "m. bedroom": "Master Bedroom",
    "m bedroom": "Master Bedroom",
    "mbr": "Master Bedroom",
    "bedroom": "Bedroom 2",
    "bed room": "Bedroom 2",
    "bedroom 1": "Master Bedroom",
    "bedroom-1": "Master Bedroom",
    "bedroom1": "Master Bedroom",
    "bedroom 2": "Bedroom 2",
    "bedroom-2": "Bedroom 2",
    "bedroom2": "Bedroom 2",
    "bedroom 3": "Bedroom 3",
    "bedroom-3": "Bedroom 3",
    "bedroom3": "Bedroom 3",
    "guest bedroom": "Guest Bedroom",
    "guest room": "Guest Bedroom",
    "kids bedroom": "Kids' Bedroom",
    "kids' bedroom": "Kids' Bedroom",
    "kid's bedroom": "Kids' Bedroom",
    "children bedroom": "Kids' Bedroom",
    "study": "Study / Home Office",
    "home office": "Study / Home Office",
    "study / home office": "Study / Home Office",
    "puja": "Puja Room",
    "puja room": "Puja Room",
    "pooja": "Puja Room",
    "pooja room": "Puja Room",
    "entertainment room": "Entertainment / Family Room",
    "family room": "Entertainment / Family Room",
    "entertainment / family room": "Entertainment / Family Room",
    "home theatre": "Home Theatre",
    "home theater": "Home Theatre",
    "bathroom": "Bathrooms / Toilets",
    "bathrooms / toilets": "Bathrooms / Toilets",
    "toilet": "Bathrooms / Toilets",
    "toilet-1": "Bathrooms / Toilets",
    "toilet-2": "Bathrooms / Toilets",
    "toilet 1": "Bathrooms / Toilets",
    "toilet 2": "Bathrooms / Toilets",
    "m. toilet": "Bathrooms / Toilets",
    "m toilet": "Bathrooms / Toilets",
    "washroom": "Bathrooms / Toilets",
    "utility": "Utility / Laundry",
    "laundry": "Utility / Laundry",
    "utility / laundry": "Utility / Laundry",
    "balcony": "Balcony",
    "m. balcony": "Balcony",
    "m balcony": "Balcony",
    "terrace": "Terrace",
    "entrance": "Entrance / Foyer",
    "foyer": "Entrance / Foyer",
    "entrance lobby": "Entrance / Foyer",
    "entrance / foyer": "Entrance / Foyer",
    "walk-in wardrobe": "Walk-in Wardrobe",
    "walk in wardrobe": "Walk-in Wardrobe",
    "dressing room": "Dressing Room",
    "dressing": "Dressing Room",
    "store": "Store Room",
    "store room": "Store Room",
    "storeroom": "Store Room",
    "servant room": "Servant / Staff Room",
    "staff room": "Servant / Staff Room",
    "servant / staff room": "Servant / Staff Room",
}

# Curated furniture names from quiz options (furniture only — no capacity/style noise)
DEFAULT_CATALOG: dict[str, list[str]] = {
    "Living Room": [
        "1-seater sofa",
        "2-seater sofa",
        "3-seater sofa",
        "L-shaped sofa",
        "U-shaped sofa",
        "Sofa + chaise",
        "Modular sofa",
        "Recliner",
        "Sofa + accent chairs",
        "Low Indian seating",
        "Accent chair",
        "Occasional chairs",
        "Poufs / ottomans",
        "Bench",
        "Coffee table",
        "Nesting tables",
        "TV unit",
        "Low console",
        "Bar cabinet",
        "Floor lamp",
        "Side table",
    ],
    "Dining Room": [
        "Rectangular dining table",
        "Square dining table",
        "Round dining table",
        "Oval dining table",
        "Extendable dining table",
        "Dining chairs",
        "Chairs + bench",
        "Crockery unit",
        "Console",
        "Bar cabinet",
        "Buffet counter",
        "Full-height storage",
        "Small bar cabinet",
        "Bar counter",
    ],
    "Kitchen": [
        "Kitchen island",
        "Peninsula counter",
        "Hob + chimney",
        "Built-in oven",
        "Freestanding cooking range",
        "Refrigerator",
        "Microwave",
        "Dishwasher",
        "Tall units",
        "Pantry",
        "Appliance garage",
        "Open shelves",
        "Washing machine",
        "Utility counter",
    ],
    "Master Bedroom": [
        "Queen size bed",
        "King size bed",
        "Double bed",
        "Platform bed",
        "Storage bed",
        "Side tables",
        "Wardrobe",
        "Walk-in wardrobe",
        "Wardrobe + dressing",
        "Dressing table",
        "Armchair",
        "Bench",
        "Lounge chair",
        "Chaise",
        "TV unit",
        "Study desk",
        "Floor lamp",
    ],
    "Bedroom 2": [
        "Queen size bed",
        "King size bed",
        "Double bed",
        "Single bed",
        "Side tables",
        "Wardrobe",
        "Dressing table",
        "Study desk",
        "Armchair",
        "TV unit",
        "Floor lamp",
    ],
    "Bedroom 3": [
        "Queen size bed",
        "King size bed",
        "Double bed",
        "Single bed",
        "Side tables",
        "Wardrobe",
        "Dressing table",
        "Study desk",
        "Armchair",
        "TV unit",
        "Floor lamp",
    ],
    "Guest Bedroom": [
        "Single bed",
        "Queen size bed",
        "King size bed",
        "Sofa bed",
        "Daybed",
        "Twin beds",
        "Bunk bed",
        "Wardrobe",
        "Bedside tables",
        "Dressing table",
        "TV unit",
        "Study desk",
    ],
    "Kids' Bedroom": [
        "Single bed",
        "Bunk bed",
        "Loft bed",
        "Trundle bed",
        "Storage bed",
        "Study desk",
        "Wardrobe",
        "Open shelves",
        "Toy storage",
        "Book storage",
        "Floor lamp",
    ],
    "Study / Home Office": [
        "Straight desk",
        "L-shaped desk",
        "U-shaped desk",
        "Wall-mounted desk",
        "Built-in desk",
        "Executive desk",
        "Two-person desk",
        "Task chair",
        "Ergonomic chair",
        "Executive chair",
        "Lounge + work combination",
        "Bookshelf",
        "Floor lamp",
    ],
    "Puja Room": [
        "Puja cabinet",
        "Wall-mounted unit",
        "Open shelves",
        "Full-height unit",
        "Corner puja unit",
    ],
    "Entertainment / Family Room": [
        "Sofa",
        "L-shaped sofa",
        "Sectional",
        "Recliners",
        "Bean bags",
        "TV unit",
        "Coffee table",
        "Floor lamp",
    ],
    "Home Theatre": [
        "Individual recliners",
        "2-seater recliner",
        "3-seater recliner",
        "Tiered seating",
        "Sofa + recliner",
        "TV unit",
        "Projector screen",
    ],
    "Bathrooms / Toilets": [
        "WC + basin",
        "WC + basin + shower",
        "WC + basin + bathtub",
        "Vanity unit",
        "Mirror cabinet",
    ],
    "Utility / Laundry": [
        "Washing machine",
        "Dryer",
        "Washer-dryer",
        "Laundry sink",
        "Utility shelves",
        "Tall storage",
        "Laundry baskets",
    ],
    "Balcony": [
        "Outdoor seating",
        "2 chairs + table",
        "Plant stands",
        "Dining set",
        "Lounge chair",
    ],
    "Terrace": [
        "Outdoor sofa",
        "Dining table",
        "Lounge seating",
        "Bar counter",
        "BBQ area",
        "Yoga / exercise zone seating",
    ],
    "Entrance / Foyer": [
        "Shoe cabinet",
        "Shoe + bench",
        "Console",
        "Mirror",
        "Pouf",
        "Chair",
        "Small bench",
    ],
    "Walk-in Wardrobe": [
        "Long hanging",
        "Short hanging",
        "Drawers",
        "Open shelves",
        "Shoe storage",
        "Island wardrobe",
        "Jewellery storage",
    ],
    "Dressing Room": [
        "Dressing table",
        "Full-height mirror",
        "Drawers",
        "Open shelves",
        "Seating stool",
    ],
    "Store Room": [
        "Shelves",
        "Cabinets",
        "Tall storage",
    ],
    "Servant / Staff Room": [
        "Single bed",
        "Wardrobe",
        "Side table",
        "Chair",
    ],
}


def default_catalog_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "furniture_catalog.json"


def build_default_catalog_payload() -> dict[str, Any]:
    return {
        "source": "QUIZ FOR REQURIEMENT_DEZINEBOX .xlsx (space + furniture names only)",
        "spaces": CANONICAL_SPACES,
        "furniture_by_space": DEFAULT_CATALOG,
        "aliases": SPACE_ALIASES,
    }


def save_catalog(path: Path | None = None) -> Path:
    path = path or default_catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_default_catalog_payload()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    path = path or default_catalog_path()
    if not path.exists():
        save_catalog(path)
    return json.loads(path.read_text(encoding="utf-8"))


def list_spaces(catalog: dict[str, Any] | None = None) -> list[str]:
    catalog = catalog or load_catalog()
    spaces = catalog.get("spaces") or list(catalog.get("furniture_by_space", {}).keys())
    return list(spaces)


def furniture_for_space(space: str, catalog: dict[str, Any] | None = None) -> list[str]:
    catalog = catalog or load_catalog()
    by_space = catalog.get("furniture_by_space", {})
    canonical = normalize_space_name(space, catalog)
    if canonical and canonical in by_space:
        return list(by_space[canonical])
    if space in by_space:
        return list(by_space[space])
    return []


def normalize_space_name(name: str, catalog: dict[str, Any] | None = None) -> str | None:
    catalog = catalog or load_catalog()
    raw = (name or "").strip()
    if not raw:
        return None
    spaces = set(list_spaces(catalog))
    if raw in spaces:
        return raw
    aliases = catalog.get("aliases") or SPACE_ALIASES
    key = re.sub(r"\s+", " ", raw.lower())
    key = key.replace("’", "'")
    if key in aliases:
        return aliases[key]
    # fuzzy contains
    for alias, canonical in aliases.items():
        if alias in key or key in alias:
            return canonical
    for space in spaces:
        if space.lower() in key or key in space.lower():
            return space
    return None


def map_detected_to_catalog(
    detected: list[str],
    catalog: dict[str, Any] | None = None,
) -> list[str]:
    catalog = catalog or load_catalog()
    mapped: list[str] = []
    seen: set[str] = set()
    for name in detected:
        canonical = normalize_space_name(name, catalog)
        if canonical and canonical not in seen:
            seen.add(canonical)
            mapped.append(canonical)
    return mapped
