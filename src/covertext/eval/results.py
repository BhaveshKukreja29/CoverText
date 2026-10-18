"""JSON results shared by the experiment runner and both benchmarks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from covertext.common.schema import StegoRecord


def write_results(
    path: str | Path,
    summary: list[dict[str, Any]],
    records: list[StegoRecord],
) -> None:
    """Write ``{"summary": ..., "records": ...}``. ``load_records`` reads the records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": _jsonable(summary),
        "records": [_jsonable(record.to_dict()) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_results(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "summary" not in data or "records" not in data:
        raise ValueError("results file must contain 'summary' and 'records'")
    return data


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
