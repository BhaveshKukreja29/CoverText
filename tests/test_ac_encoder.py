import pytest

from covertext.common.interfaces import Encoder
from covertext.common.payload import generate_payload
from covertext.encoder.ac import ACEncoder


@pytest.mark.slow
def test_ac_roundtrip_8bit(model_fixture):
    _assert_roundtrip(model_fixture, 8, seed=0)


@pytest.mark.slow
def test_ac_roundtrip_16bit(model_fixture):
    _assert_roundtrip(model_fixture, 16, seed=1)


@pytest.mark.slow
def test_ac_roundtrip_32bit(model_fixture):
    _assert_roundtrip(model_fixture, 32, seed=2)


@pytest.mark.slow
def test_ac_roundtrip_64bit(model_fixture):
    _assert_roundtrip(model_fixture, 64, seed=3)


@pytest.mark.slow
def test_ac_roundtrip_200_payloads(model_fixture):
    model, tokenizer = model_fixture
    encoder = ACEncoder(model, tokenizer)
    context = "The history of science is"
    trial = 0
    for size in (8, 16, 32, 64):
        for _ in range(50):
            payload = generate_payload(size, seed=1000 + trial)
            stego = encoder.encode(payload, context)
            recovered = encoder.decode(stego, context, size)
            assert recovered == payload, f"trial {trial} size {size}"
            trial += 1


@pytest.mark.slow
def test_ac_stego_text_is_decodable_string(model_fixture):
    model, tokenizer = model_fixture
    encoder = ACEncoder(model, tokenizer)
    stego = encoder.encode("01001101", "Once upon a time")
    assert isinstance(stego, str)
    assert len(stego) > 0


def test_ac_implements_interface():
    dummy_model = type("M", (), {"device": "cpu"})()
    encoder = ACEncoder(dummy_model, tokenizer=object())
    assert isinstance(encoder, Encoder)


def _assert_roundtrip(model_fixture, num_bits: int, seed: int) -> None:
    model, tokenizer = model_fixture
    encoder = ACEncoder(model, tokenizer)
    payload = generate_payload(num_bits, seed=seed)
    context = "The capital of France is"
    stego = encoder.encode(payload, context)
    recovered = encoder.decode(stego, context, num_bits)
    assert recovered == payload
