import pytest

from covertext.common.data import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_TOKENS,
    clean_paragraphs,
    get_paragraph_ids,
    get_splits,
    load_wikitext103,
)
from covertext.common.model import MODEL_ID


@pytest.mark.slow
def test_deterministic_splits():
    first = get_splits(seed=42)
    second = get_splits(seed=42)
    assert first == second


@pytest.mark.slow
def test_clean_paragraphs_no_headers():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    raw = load_wikitext103()
    for split in ("train", "validation", "test"):
        paragraphs = clean_paragraphs(raw[split], tokenizer)
        assert all(not p.startswith(" = ") and not p.lstrip().startswith("=") for p in paragraphs)


@pytest.mark.slow
def test_paragraph_token_length():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    train, val, test = get_splits(tokenizer=tokenizer)
    for paragraphs in (train, val, test):
        for paragraph in paragraphs:
            n = len(tokenizer.encode(paragraph, add_special_tokens=False))
            assert DEFAULT_MIN_TOKENS <= n <= DEFAULT_MAX_TOKENS


@pytest.mark.slow
def test_paragraph_ids_deterministic():
    paragraphs = ["a", "b", "c"]
    assert get_paragraph_ids(paragraphs, seed=42) == get_paragraph_ids(paragraphs, seed=42)
