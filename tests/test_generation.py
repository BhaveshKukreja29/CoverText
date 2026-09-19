import pytest

from covertext.common.generation import reversible_subset


class _SelectiveTokenizer:
    """Round-trips every token except 7."""

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)

    def encode(self, text, add_special_tokens=False):
        ids = [ord(c) for c in text]
        if 7 in ids:
            return [0]
        return ids


def test_reversible_subset_drops_non_roundtrip_tokens():
    probs, ids = reversible_subset(
        _SelectiveTokenizer(), [], [0.4, 0.4, 0.2], [10, 7, 11]
    )
    assert ids == [10, 11]
    assert pytest.approx(sum(probs)) == 1.0


def test_reversible_subset_single_token_is_not_a_raw_fallback():
    probs, ids = reversible_subset(_SelectiveTokenizer(), [], [0.7, 0.3], [7, 10])
    assert ids == [10]
    assert probs == [1.0]


def test_reversible_subset_raises_when_none_roundtrip():
    with pytest.raises(ValueError, match="no reversible tokens"):
        reversible_subset(_SelectiveTokenizer(), [], [1.0], [7])
