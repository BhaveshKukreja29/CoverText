"""Stego record schema — the shared data format consumed by all tracks."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StegoRecord:
    """One steganographic encoding instance."""

    cover_text: str
    stego_text: str
    payload_bits: str
    method_name: str
    model_name: str
    generation_settings: dict[str, Any] = field(default_factory=dict)
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StegoRecord:
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> StegoRecord:
        return cls.from_dict(json.loads(s))


def save_records(records: list[StegoRecord], path: str | Path) -> None:
    """Write a JSON array of records to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.to_dict() for record in records]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_records(path: str | Path) -> list[StegoRecord]:
    """Read a JSON array from a file and return a list of StegoRecord."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [StegoRecord.from_dict(item) for item in data]
