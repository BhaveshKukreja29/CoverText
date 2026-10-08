#!/usr/bin/env python3
"""Build and save the held-out clean corpus; verify no overlap with encoder contexts."""

from covertext.detector.corpus import (
    build_clean_corpus,
    get_encoder_contexts,
    save_corpus,
    verify_no_overlap,
)


def main() -> None:
    clean = build_clean_corpus()
    encoder_contexts = get_encoder_contexts()
    disjoint = verify_no_overlap(clean, encoder_contexts)
    save_corpus(clean, "data/clean_corpus.json")
    print(f"overlap_check={disjoint}")
    print(f"clean_corpus_size={len(clean)}")
    print(f"encoder_contexts_size={len(encoder_contexts)}")
    print("saved=data/clean_corpus.json")


if __name__ == "__main__":
    main()
