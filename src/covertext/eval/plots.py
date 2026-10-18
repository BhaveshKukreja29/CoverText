"""Charts from an experiment summary.

The headline figure plots capacity against detector AUROC, one line per
encoder and detector pair, with points joined in payload-size order. Decode
accuracy and fluency are encoder properties, so those figures use one line
per encoder. When several detector rows share an encoder and a payload size,
the plotted value is their arithmetic mean. Fluency rows that are null,
because a continuation was too short for a conditional perplexity, are left
out of that mean.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAPACITY_AUROC = "capacity_auroc.png"
DECODE_ACCURACY = "decode_accuracy.png"
FLUENCY_LOSS = "fluency_loss.png"

_REQUIRED = (
    "encoder",
    "payload_bits",
    "capacity_bpt",
    "decode_accuracy_bit",
    "fluency_loss",
    "auroc",
)


def capacity_auroc_series(summary: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    """``'AC / PPL'`` to ``(capacity_bpt, auroc)`` in increasing payload size."""
    _require_rows(summary)
    buckets: dict[tuple[str, str], dict[int, list[tuple[float, float]]]] = {}
    for row in summary:
        if row["auroc"] is None:
            continue
        key = (str(row["encoder"]), str(row["detector"]))
        size = int(row["payload_bits"])
        buckets.setdefault(key, {}).setdefault(size, []).append(
            (float(row["capacity_bpt"]), float(row["auroc"]))
        )
    series: dict[str, list[tuple[float, float]]] = {}
    for key in sorted(buckets):
        label = f"{key[0]} / {key[1]}" if key[1] not in {"", "None"} else key[0]
        points = []
        for size in sorted(buckets[key]):
            pairs = buckets[key][size]
            points.append(
                (
                    statistics.fmean(item[0] for item in pairs),
                    statistics.fmean(item[1] for item in pairs),
                )
            )
        series[label] = points
    return series


def decode_accuracy_series(summary: list[dict[str, Any]]) -> dict[str, list[tuple[int, float]]]:
    """Encoder name to ``(payload_bits, bit accuracy)`` in increasing payload size."""
    return _encoder_metric(summary, "decode_accuracy_bit")


def fluency_series(summary: list[dict[str, Any]]) -> dict[str, list[tuple[int, float]]]:
    """Encoder name to ``(payload_bits, fluency loss)`` where the loss is finite."""
    return _encoder_metric(summary, "fluency_loss")


def plot_summary(summary: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Path]:
    """Render the three phase-one charts. Returns their paths."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "capacity_auroc": output / CAPACITY_AUROC,
        "decode_accuracy": output / DECODE_ACCURACY,
        "fluency_loss": output / FLUENCY_LOSS,
    }
    _draw(
        paths["capacity_auroc"],
        capacity_auroc_series(summary),
        "Capacity (bits per token)",
        "Detector AUROC",
        "Capacity against detector AUROC",
    )
    _draw(
        paths["decode_accuracy"],
        decode_accuracy_series(summary),
        "Payload size (bits)",
        "Bit accuracy",
        "Decode accuracy against payload size",
    )
    _draw(
        paths["fluency_loss"],
        fluency_series(summary),
        "Payload size (bits)",
        "Perplexity delta",
        "Fluency loss against payload size",
    )
    return paths


def _encoder_metric(summary: list[dict[str, Any]], field: str) -> dict[str, list[tuple[int, float]]]:
    _require_rows(summary)
    buckets: dict[str, dict[int, list[float]]] = {}
    for row in summary:
        value = row[field]
        if value is None:
            continue
        number = float(value)
        encoder = str(row["encoder"])
        size = int(row["payload_bits"])
        buckets.setdefault(encoder, {}).setdefault(size, []).append(number)
    series: dict[str, list[tuple[int, float]]] = {}
    for encoder in sorted(buckets):
        series[encoder] = [
            (size, statistics.fmean(buckets[encoder][size])) for size in sorted(buckets[encoder])
        ]
    return series


def _require_rows(summary: list[dict[str, Any]]) -> None:
    if not summary:
        raise ValueError("summary is empty")
    missing = [key for key in _REQUIRED if key not in summary[0]]
    if missing:
        raise ValueError(f"summary row is missing {missing}")


def _draw(path: Path, series: dict, xlabel: str, ylabel: str, title: str) -> None:
    figure, axes = plt.subplots(figsize=(6.5, 4.5))
    for label, points in series.items():
        axes.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label=label,
        )
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    if series:
        axes.legend()
    else:
        axes.text(0.5, 0.5, "no finite values", ha="center", va="center", transform=axes.transAxes)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
