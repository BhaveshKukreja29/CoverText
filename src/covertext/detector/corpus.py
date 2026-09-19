"""Held-out clean corpus for the detector track."""

from __future__ import annotations

import json
from pathlib import Path

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from covertext.common.data import CACHE_DIR, get_splits


def build_clean_corpus(
    cache_dir: str | Path = CACHE_DIR,
    tokenizer: PreTrainedTokenizerFast | None = None,
    min_tokens: int = 64,
    max_tokens: int = 256,
) -> list[str]:
    """Return the WikiText-103 test split as the detector's clean corpus."""
    train, _, test = get_splits(
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    encoder_set = set(train)
    return [paragraph for paragraph in test if paragraph not in encoder_set]


def get_encoder_contexts(
    cache_dir: str | Path = CACHE_DIR,
    tokenizer: PreTrainedTokenizerFast | None = None,
    min_tokens: int = 64,
    max_tokens: int = 256,
) -> list[str]:
    """Return the WikiText-103 train split as encoder cover/context text."""
    train, _, _ = get_splits(
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    return train


def verify_no_overlap(clean_corpus: list[str], encoder_contexts: list[str]) -> bool:
    """Return True if the two sets are disjoint; raise ValueError otherwise."""
    if set(clean_corpus).isdisjoint(set(encoder_contexts)):
        return True
    raise ValueError("clean corpus overlaps encoder contexts")


def save_corpus(paragraphs: list[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(paragraphs, indent=2), encoding="utf-8")


def load_corpus(path: str | Path) -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
