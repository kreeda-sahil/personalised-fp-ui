"""Estimated OpenAI API costing for the POC."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# USD rates — tweak as needed for the POC
GPT4O_INPUT_PER_1M = 2.50
GPT4O_OUTPUT_PER_1M = 10.00
# Assumed gpt-image-2 unit price (estimated; label as estimate in UI)
GPT_IMAGE_2_PER_IMAGE = 0.08

USD_TO_INR = 95.0


@dataclass
class CostEntry:
    label: str
    model: str
    call_type: str  # vision | image
    estimated_usd: float
    details: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    images: int = 0

    @property
    def estimated_inr(self) -> float:
        return round(self.estimated_usd * USD_TO_INR, 2)


@dataclass
class CostTracker:
    entries: list[CostEntry] = field(default_factory=list)

    def add(self, entry: CostEntry) -> CostEntry:
        self.entries.append(entry)
        return entry

    def add_vision(
        self,
        *,
        label: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostEntry:
        usd = (input_tokens / 1_000_000) * GPT4O_INPUT_PER_1M + (
            output_tokens / 1_000_000
        ) * GPT4O_OUTPUT_PER_1M
        return self.add(
            CostEntry(
                label=label,
                model=model,
                call_type="vision",
                estimated_usd=round(usd, 6),
                details=f"{input_tokens} in / {output_tokens} out tokens",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )

    def add_image(
        self,
        *,
        label: str,
        model: str,
        images: int = 1,
        unit_price: float | None = None,
        details: str = "",
    ) -> CostEntry:
        price = unit_price if unit_price is not None else GPT_IMAGE_2_PER_IMAGE
        usd = price * images
        return self.add(
            CostEntry(
                label=label,
                model=model,
                call_type="image",
                estimated_usd=round(usd, 6),
                details=details or f"{images} image(s) × ${price:.4f}",
                images=images,
            )
        )

    @property
    def total_usd(self) -> float:
        return round(sum(e.estimated_usd for e in self.entries), 6)

    @property
    def total_inr(self) -> float:
        return round(self.total_usd * USD_TO_INR, 2)

    def clear(self) -> None:
        self.entries.clear()

    def to_dicts(self) -> list[dict[str, Any]]:
        rows = []
        for e in self.entries:
            d = asdict(e)
            d["estimated_inr"] = e.estimated_inr
            rows.append(d)
        return rows

    @classmethod
    def from_dicts(cls, rows: list[dict[str, Any]] | None) -> "CostTracker":
        tracker = cls()
        if not rows:
            return tracker
        for row in rows:
            tracker.entries.append(
                CostEntry(
                    label=row.get("label", ""),
                    model=row.get("model", ""),
                    call_type=row.get("call_type", ""),
                    estimated_usd=float(row.get("estimated_usd", 0)),
                    details=row.get("details", ""),
                    input_tokens=int(row.get("input_tokens", 0) or 0),
                    output_tokens=int(row.get("output_tokens", 0) or 0),
                    images=int(row.get("images", 0) or 0),
                )
            )
        return tracker
