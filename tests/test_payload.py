from covertext.common.payload import (
    bits_to_bytes,
    bits_to_hex,
    bytes_to_bits,
    generate_payload,
    hex_to_bits,
)


def test_deterministic_same_seed():
    assert generate_payload(64, seed=42) == generate_payload(64, seed=42)


def test_different_seeds_no_collision():
    payloads = {generate_payload(64, seed=i) for i in range(5000)}
    assert len(payloads) == 5000


def test_correct_length():
    for num_bits in (8, 16, 32, 64, 128):
        assert len(generate_payload(num_bits, seed=0)) == num_bits


def test_bits_to_bytes_roundtrip():
    for bits in ("0", "1", "01001000", "01101", "11111111", "00000000", "101010"):
        recovered = bytes_to_bits(bits_to_bytes(bits))
        assert recovered.startswith(bits)


def test_bits_hex_roundtrip():
    for bits in ("01001000", "1111", "0000", "1010", "11001100"):
        assert hex_to_bits(bits_to_hex(bits)) == bits


def test_bits_to_bytes_example():
    assert bits_to_bytes("01001000") == b"H"


def test_bytes_to_bits_example():
    assert bytes_to_bits(b"H") == "01001000"


def test_bits_to_hex_example():
    assert bits_to_hex("01001000") == "48"


def test_bytes_roundtrip_with_bit_length():
    for bits in ("0", "1", "01101", "101010", "01001000"):
        assert bytes_to_bits(bits_to_bytes(bits), bit_length=len(bits)) == bits


def test_hex_roundtrip_with_bit_length():
    for bits in ("1", "01", "011", "01001000"):
        assert hex_to_bits(bits_to_hex(bits), bit_length=len(bits)) == bits
