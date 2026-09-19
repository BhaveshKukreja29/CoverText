import pytest

from covertext.detector.corpus import (
    build_clean_corpus,
    get_encoder_contexts,
    load_corpus,
    save_corpus,
    verify_no_overlap,
)


@pytest.mark.slow
def test_no_overlap():
    clean = build_clean_corpus()
    encoder_contexts = get_encoder_contexts()
    assert verify_no_overlap(clean, encoder_contexts) is True


@pytest.mark.slow
def test_corpus_not_empty():
    assert len(build_clean_corpus()) > 0


def test_save_load_roundtrip(tmp_path):
    paragraphs = ["alpha paragraph", "beta paragraph"]
    path = tmp_path / "corpus.json"
    save_corpus(paragraphs, path)
    assert load_corpus(path) == paragraphs
