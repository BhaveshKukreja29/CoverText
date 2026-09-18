import numpy as np

from covertext.encoder.arithmetic import ArithmeticCoder


def _repeat(dist, n: int = 2048):
    return [dist] * n


def _roundtrip(bits: str, dist, precision: int = 32) -> str:
    coder = ArithmeticCoder(precision=precision)
    seq = _repeat(dist)
    symbols = coder.encode(bits, seq)
    return coder.decode(symbols, seq, len(bits))


def test_roundtrip_uniform_distribution():
    dist = [0.25, 0.25, 0.25, 0.25]
    rng = np.random.default_rng(0)
    for seed in range(100):
        bits = "".join(str(int(b)) for b in rng.integers(0, 2, size=64))
        assert _roundtrip(bits, dist) == bits


def test_roundtrip_nonuniform_distribution():
    dist = [0.7, 0.1, 0.1, 0.1]
    rng = np.random.default_rng(1)
    for _ in range(100):
        bits = "".join(str(int(b)) for b in rng.integers(0, 2, size=64))
        assert _roundtrip(bits, dist) == bits


def test_roundtrip_many_trials():
    rng = np.random.default_rng(2)
    lengths = [8, 16, 32, 64]
    for _ in range(1000):
        n = int(rng.integers(2, 17))
        dist = rng.dirichlet(np.ones(n)).tolist()
        num_bits = int(rng.choice(lengths))
        bits = "".join(str(int(b)) for b in rng.integers(0, 2, size=num_bits))
        assert _roundtrip(bits, dist) == bits


def test_roundtrip_varying_vocab_sizes():
    rng = np.random.default_rng(3)
    bits = "".join(str(int(b)) for b in rng.integers(0, 2, size=32))
    for vocab in (2, 10, 100, 1000):
        dist = np.ones(vocab) / vocab
        assert _roundtrip(bits, dist) == bits


def test_single_bit_payload():
    dist = [0.5, 0.5]
    assert _roundtrip("0", dist) == "0"
    assert _roundtrip("1", dist) == "1"


def test_empty_payload():
    coder = ArithmeticCoder()
    dist = [0.5, 0.5]
    symbols = coder.encode("", _repeat(dist))
    assert symbols == []
    assert coder.decode(symbols, _repeat(dist), 0) == ""
