"""Deterministic payload bitstring generation and conversions."""

from __future__ import annotations

import random


def generate_payload(num_bits: int, seed: int | None = None) -> str:
    """Return a string of ``'0'``/``'1'`` characters of length ``num_bits``."""
    if num_bits < 0:
        raise ValueError("num_bits must be non-negative")
    rng = random.Random(seed)
    return "".join(str(rng.getrandbits(1)) for _ in range(num_bits))


def bits_to_bytes(bits: str) -> bytes:
    """Convert a bitstring to bytes, padding with trailing zeros to a multiple of 8."""
    if not bits:
        return b""
    if any(c not in "01" for c in bits):
        raise ValueError("bits must contain only '0' and '1'")
    padded = bits + "0" * ((8 - len(bits) % 8) % 8)
    return bytes(int(padded[i : i + 8], 2) for i in range(0, len(padded), 8))


def bytes_to_bits(data: bytes) -> str:
    """Convert bytes to a bitstring."""
    return "".join(f"{byte:08b}" for byte in data)


def bits_to_hex(bits: str) -> str:
    """Convert a bitstring to a hexadecimal string, padding to a multiple of 4 bits."""
    if not bits:
        return ""
    if any(c not in "01" for c in bits):
        raise ValueError("bits must contain only '0' and '1'")
    padded = bits + "0" * ((4 - len(bits) % 4) % 4)
    width = len(padded) // 4
    return f"{int(padded, 2):0{width}x}"


def hex_to_bits(hex_str: str) -> str:
    """Convert a hexadecimal string back to a bitstring (4 bits per hex digit)."""
    if not hex_str:
        return ""
    n_bits = len(hex_str) * 4
    return f"{int(hex_str, 16):0{n_bits}b}"
