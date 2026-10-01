"""Regenerate data/furniture_catalog.json from curated catalog defaults.

The quiz Excel is the source of space/furniture naming; this script writes the
stable JSON snapshot used by the Streamlit UI.
"""

from __future__ import annotations

from src.catalog import save_catalog


if __name__ == "__main__":
    path = save_catalog()
    print(f"Wrote {path}")
