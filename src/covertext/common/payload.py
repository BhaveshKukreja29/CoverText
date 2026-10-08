"""Deterministic payload bitstring generation and conversions.

``bits_to_bytes`` / ``bits_to_hex`` pad to a multiple of 8 or 4 bits. The inverse
functions take an optional ``bit_length`` so the original string can be recovered
exactly when that length is known. Without it, padded zeros remain.
"""

from __future__ import annotations

import random


def generate_payload(num_bits: int, seed: int | None = None) -> str:
    """Return a string of ``'0'``/``'1'`` characters of length ``num_bits``."""
    if num_bits < 0:
        raise ValueError("num_bits must be non-negative")
    rng = random.Random(seed)
    return "".join(str(rng.getrandbits(1)) for _ in range(num_bits))


def _require_bits(bits: str) -> None:
    if any(c not in "01" for c in bits):
        raise ValueError("bits must contain only '0' and '1'")


def bits_to_bytes(bits: str) -> bytes:
    """Convert a bitstring to bytes, padding with trailing zeros to a multiple of 8."""
    if not bits:
        return b""
    _require_bits(bits)
    padded = bits + "0" * ((8 - len(bits) % 8) % 8)
    return bytes(int(padded[i : i + 8], 2) for i in range(0, len(padded), 8))


def bytes_to_bits(data: bytes, bit_length: int | None = None) -> str:
    """Convert bytes to a bitstring.

    If ``bit_length`` is given, the result is truncated to that many bits so a
    padded ``bits_to_bytes`` round-trip is lossless.
    """
    bits = "".join(f"{byte:08b}" for byte in data)
    if bit_length is None:
        return bits
    if bit_length < 0 or bit_length > len(bits):
        raise ValueError("bit_length must be between 0 and 8 * len(data)")
    return bits[:bit_length]


def bits_to_hex(bits: str) -> str:
    """Convert a bitstring to hex, padding with trailing zeros to a multiple of 4."""
    if not bits:
        return ""
    _require_bits(bits)
    padded = bits + "0" * ((4 - len(bits) % 4) % 4)
    width = len(padded) // 4
    return f"{int(padded, 2):0{width}x}"


def hex_to_bits(hex_str: str, bit_length: int | None = None) -> str:
    """Convert hex to a bitstring (4 bits per digit).

    If ``bit_length`` is given, truncate to that length after decoding.
    """
    if not hex_str:
        return ""
    n_bits = len(hex_str) * 4
    bits = f"{int(hex_str, 16):0{n_bits}b}"
    if bit_length is None:
        return bits
    if bit_length < 0 or bit_length > len(bits):
        raise ValueError("bit_length must be between 0 and 4 * len(hex_str)")
    return bits[:bit_length]
