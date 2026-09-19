import pytest

from covertext.common.interfaces import Encoder
from covertext.common.payload import generate_payload
import numpy as np

from covertext.encoder.mec_encoder import MECEncoder


@pytest.mark.slow
def test_mec_roundtrip_8bit(model_fixture):
    _assert_roundtrip(model_fixture, 8, seed=0)


@pytest.mark.slow
def test_mec_roundtrip_16bit(model_fixture):
    _assert_roundtrip(model_fixture, 16, seed=1)


@pytest.mark.slow
def test_mec_roundtrip_32bit(model_fixture):
    _assert_roundtrip(model_fixture, 32, seed=2)


@pytest.mark.slow
def test_mec_roundtrip_64bit(model_fixture):
    _assert_roundtrip(model_fixture, 64, seed=3)


@pytest.mark.slow
def test_mec_roundtrip_200_payloads(model_fixture):
    model, tokenizer = model_fixture
    encoder = MECEncoder(model, tokenizer)
    context = "The history of science is"
    trial = 0
    for size in (8, 16, 32, 64):
        for _ in range(50):
            payload = generate_payload(size, seed=2000 + trial)
            stego = encoder.encode(payload, context)
            recovered = encoder.decode(stego, context, size)
            assert recovered == payload, f"trial {trial} size {size}"
            trial += 1


def test_mec_implements_interface():
    dummy_model = type("M", (), {"device": "cpu"})()
    encoder = MECEncoder(dummy_model, tokenizer=object())
    assert isinstance(encoder, Encoder)


def test_bits_per_step_rejects_split_rows():
    encoder = MECEncoder(type("M", (), {"device": "cpu"})(), tokenizer=object())
    # review2 counterexample: p_max <= 0.5 so n=1 looked legal, but token 2 splits
    probs = [0.4, 0.3, 0.3]
    assert encoder._bits_per_step(probs, 8) == 0
    coupling = encoder._coupling(probs, 1)
    assert int(np.sum(coupling > 1e-12, axis=1).max()) > 1


def test_coupling_sample_inverts():
    encoder = MECEncoder(type("M", (), {"device": "cpu"})(), tokenizer=object(), seed=7)
    probs = [0.4, 0.3, 0.2, 0.1]
    n_bits = encoder._bits_per_step(probs, 8)
    assert n_bits >= 1
    for message in range(1 << n_bits):
        for step in range(20):
            token = encoder._sample_token_index(probs, message, n_bits, step)
            recovered = encoder._recover_message(probs, token, n_bits)
            assert recovered == message


def test_mec_token_marginal_tracks_model():
    encoder = MECEncoder(type("M", (), {"device": "cpu"})(), tokenizer=object(), seed=3)
    probs = [0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.04]
    n_bits = encoder._bits_per_step(probs, 8)
    n_messages = 1 << n_bits
    counts = np.zeros(len(probs))
    for step in range(3000):
        message = step % n_messages
        counts[encoder._sample_token_index(probs, message, n_bits, step)] += 1
    empirical = counts / counts.sum()
    uniform = np.full(len(probs), 1.0 / len(probs))
    assert np.sum(np.abs(empirical - np.array(probs))) < np.sum(np.abs(empirical - uniform))


def test_mec_raises_when_max_tokens_exhausted():
    encoder = MECEncoder(type("M", (), {"device": "cpu"})(), tokenizer=_FakeTokenizer())
    encoder._tokenize = lambda text: __import__("torch").tensor([[1]])

    def _next(*_args, **_kwargs):
        return [0.4, 0.3, 0.2, 0.1], [10, 11, 12, 13], None

    import covertext.encoder.mec_encoder as mec_mod

    original = mec_mod.next_token_topk
    mec_mod.next_token_topk = lambda *a, **k: _next()
    mec_mod.reversible_subset = lambda tok, gen, p, ids: (p, ids)
    try:
        with pytest.raises(RuntimeError, match="max_tokens"):
            encoder.encode("1" * 64, "ctx", max_tokens=1)
    finally:
        mec_mod.next_token_topk = original


class _FakeTokenizer:
    eos_token_id = None
    pad_token_id = None
    bos_token_id = None

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def _assert_roundtrip(model_fixture, num_bits: int, seed: int) -> None:
    model, tokenizer = model_fixture
    encoder = MECEncoder(model, tokenizer)
    payload = generate_payload(num_bits, seed=seed)
    context = "The capital of France is"
    stego = encoder.encode(payload, context)
    recovered = encoder.decode(stego, context, num_bits)
    assert recovered == payload
