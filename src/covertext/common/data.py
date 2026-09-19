"""WikiText-103 loader, paragraph cleaning, and deterministic splits."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from covertext.common.model import MODEL_ID

DATASET_NAME: str = "Salesforce/wikitext"
DATASET_CONFIG: str = "wikitext-103-v1"
DEFAULT_SEED: int = 42
DEFAULT_MIN_TOKENS: int = 64
DEFAULT_MAX_TOKENS: int = 256
CACHE_DIR: str = "data/wikitext103"
_TOKENIZE_BATCH: int = 512


def _resolve_cache_dir(cache_dir: str | Path) -> Path:
    path = Path(cache_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_header(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("=") or line.startswith(" = ")


def load_wikitext103(cache_dir: str | Path = CACHE_DIR) -> DatasetDict:
    """Download (or load cached) WikiText-103 and return the raw DatasetDict."""
    cache = _resolve_cache_dir(cache_dir)
    return load_dataset(DATASET_NAME, DATASET_CONFIG, cache_dir=str(cache))


def clean_paragraphs(
    raw_dataset: Dataset,
    tokenizer: PreTrainedTokenizerFast,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[str]:
    """Drop headers/empties, join lines into paragraphs, keep token-length window."""
    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            paragraphs.append(" ".join(buffer))
            buffer.clear()

    for line in raw_dataset["text"]:
        if not line or not line.strip() or _is_header(line):
            flush()
            continue
        buffer.append(line.strip())
    flush()

    # Characters are a cheap lower/upper bound before tokenization.
    candidates = [
        p
        for p in paragraphs
        if min_tokens <= len(p) <= max_tokens * 16
    ]
    kept: list[str] = []
    for start in range(0, len(candidates), _TOKENIZE_BATCH):
        batch = candidates[start : start + _TOKENIZE_BATCH]
        encoded = tokenizer(batch, add_special_tokens=False, truncation=False)
        for paragraph, ids in zip(batch, encoded["input_ids"]):
            if min_tokens <= len(ids) <= max_tokens:
                kept.append(paragraph)
    return kept


def get_splits(
    cache_dir: str | Path = CACHE_DIR,
    tokenizer: PreTrainedTokenizerFast | None = None,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    seed: int = DEFAULT_SEED,
) -> tuple[list[str], list[str], list[str]]:
    """Return cleaned (train, validation, test) paragraph lists."""
    del seed  # HuggingFace splits are already fixed; kept for the public API.
    cache = _resolve_cache_dir(cache_dir)
    cleaned_path = cache / f"cleaned_{min_tokens}_{max_tokens}.json"
    if cleaned_path.exists():
        payload = json.loads(cleaned_path.read_text(encoding="utf-8"))
        return payload["train"], payload["validation"], payload["test"]
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    raw = load_wikitext103(cache_dir)
    train = clean_paragraphs(raw["train"], tokenizer, min_tokens, max_tokens)
    val = clean_paragraphs(raw["validation"], tokenizer, min_tokens, max_tokens)
    test = clean_paragraphs(raw["test"], tokenizer, min_tokens, max_tokens)
    cleaned_path.write_text(
        json.dumps({"train": train, "validation": val, "test": test}),
        encoding="utf-8",
    )
    return train, val, test


def get_paragraph_ids(paragraphs: list[str], seed: int = DEFAULT_SEED) -> list[int]:
    """Deterministic paragraph ids: ``range(len(paragraphs))``."""
    del seed
    return list(range(len(paragraphs)))
