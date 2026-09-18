"""Arithmetic coder over arbitrary discrete distributions.

Interval arithmetic uses Python integers. Symbol masses are a fixed-sum
frequency table mapped onto the current interval by integer division, matching
the classic range-coder update:

    new_low  = low + width * cdf[s]     // total
    new_high = low + width * cdf[s + 1] // total
"""

from __future__ import annotations

import numpy as np

DEFAULT_PRECISION: int = 32
FREQ_TOTAL: int = 1 << 20


def _as_probs(probs: list[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if p.size == 0:
        raise ValueError("probability distribution must be non-empty")
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total <= 0.0:
        raise ValueError("probabilities must sum to a positive value")
    return p / total


def _frequencies(probs: np.ndarray) -> tuple[list[int], int]:
    """Return cdf of length n+1 and the frequency total."""
    n = probs.size
    raw = probs * FREQ_TOTAL
    counts = [int(np.floor(x)) for x in raw]
    leftover = FREQ_TOTAL - sum(counts)
    remainders = raw - np.floor(raw)
    order = list(np.argsort(-remainders, kind="stable"))
    for idx in order:
        if leftover <= 0:
            break
        if probs[idx] > 0.0:
            counts[int(idx)] += 1
            leftover -= 1
    if leftover > 0:
        positive = [i for i, p in enumerate(probs) if p > 0.0]
        counts[positive[-1]] += leftover
    cdf = [0]
    running = 0
    for c in counts:
        running += c
        cdf.append(running)
    return cdf, cdf[-1]


def _interval_bounds(low: int, high: int, cdf: list[int], total: int) -> list[tuple[int, int]]:
    """Map frequency CDF onto [low, high) with integer masses that sum to width."""
    width = high - low
    n = len(cdf) - 1
    counts = [width * (cdf[s + 1] - cdf[s]) // total for s in range(n)]
    leftover = width - sum(counts)
    remainders = sorted(
        range(n),
        key=lambda s: (width * (cdf[s + 1] - cdf[s]) % total, -s),
        reverse=True,
    )
    for s in remainders:
        if leftover <= 0:
            break
        if cdf[s + 1] > cdf[s]:
            counts[s] += 1
            leftover -= 1
    if leftover > 0:
        for s in remainders:
            counts[s] += leftover
            leftover = 0
            break
    if width > 1 and max(counts) == width:
        donor = max(range(n), key=lambda s: counts[s])
        taker = max(range(n), key=lambda s: (s != donor, cdf[s + 1] - cdf[s], -s))
        if taker != donor:
            counts[donor] -= 1
            counts[taker] += 1
    bounds = []
    cursor = low
    for c in counts:
        bounds.append((cursor, cursor + c))
        cursor += c
    return bounds


def _prefix_unique(low: int, high: int, bit_len: int, scale_bits: int) -> bool:
    if bit_len == 0 or high <= low:
        return True
    shift = scale_bits - bit_len
    if shift <= 0:
        return True
    return (low >> shift) == ((high - 1) >> shift)


def _bits_from_interval(low: int, bit_len: int, scale_bits: int) -> str:
    if bit_len == 0:
        return ""
    shift = max(scale_bits - bit_len, 0)
    value = low >> shift
    return format(value, f"0{bit_len}b")[-bit_len:]


class EncodeStream:
    def __init__(self, coder: ArithmeticCoder, bits: str):
        self.coder = coder
        self.bits = bits
        self.n_bits = len(bits)
        self.scale_bits = max(self.n_bits + coder.precision, coder.precision + 8)
        self.low = 0
        self.high = 1 << self.scale_bits
        padded = bits.ljust(self.scale_bits, "0")
        self.value = int(padded, 2) if padded else 0
        self.finished = bits == ""

    def step(self, probs: list[float] | np.ndarray) -> int:
        p = _as_probs(probs)
        cdf, total = _frequencies(p)
        bounds = _interval_bounds(self.low, self.high, cdf, total)
        symbol = len(bounds) - 1
        for idx, (start, end) in enumerate(bounds):
            if start <= self.value < end:
                symbol = idx
                break
        new_low, new_high = bounds[symbol]
        if new_high <= new_low:
            new_high = new_low + 1
        self.low, self.high = new_low, new_high
        if self.high - self.low <= 1 or (
            self.low <= self.value < self.high
            and _prefix_unique(self.low, self.high, self.n_bits, self.scale_bits)
        ):
            self.finished = True
        return symbol


class DecodeStream:
    def __init__(self, coder: ArithmeticCoder, num_bits: int | None = None):
        self.coder = coder
        self.num_bits = num_bits
        self.scale_bits: int | None = None
        self.low = 0
        self.high: int | None = None
        self.out: list[str] = []

    def _ensure_scale(self, num_bits: int) -> None:
        if self.high is not None:
            return
        self.num_bits = num_bits
        self.scale_bits = max(num_bits + self.coder.precision, self.coder.precision + 8)
        self.high = 1 << self.scale_bits

    def step(self, symbol: int, probs: list[float] | np.ndarray, final: bool = False) -> None:
        del final
        n_bits = 0 if self.num_bits is None else self.num_bits
        self._ensure_scale(n_bits)
        p = _as_probs(probs)
        cdf, total = _frequencies(p)
        if symbol < 0 or symbol >= len(cdf) - 1:
            raise ValueError(f"symbol {symbol} is outside the distribution")
        bounds = _interval_bounds(self.low, self.high, cdf, total)
        new_low, new_high = bounds[symbol]
        if new_high <= new_low:
            new_high = new_low + 1
        self.low, self.high = new_low, new_high

    def finish(self, num_bits: int) -> str:
        if num_bits == 0:
            return ""
        self._ensure_scale(num_bits)
        assert self.scale_bits is not None
        return _bits_from_interval(self.low, num_bits, self.scale_bits)


class ArithmeticCoder:
    """Arithmetic coder over caller-supplied discrete distributions."""

    def __init__(self, precision: int = DEFAULT_PRECISION):
        if precision < 2:
            raise ValueError("precision must be at least 2")
        self.precision = precision
        self.whole = 1 << precision
        self.half = self.whole >> 1
        self.quarter = self.whole >> 2

    def start_encode(self, bits: str) -> EncodeStream:
        if any(c not in "01" for c in bits):
            raise ValueError("bits must contain only '0' and '1'")
        return EncodeStream(self, bits)

    def start_decode(self, num_bits: int | None = None) -> DecodeStream:
        return DecodeStream(self, num_bits=num_bits)

    def encode(self, bits: str, probs_sequence: list[list[float] | np.ndarray]) -> list[int]:
        stream = self.start_encode(bits)
        if bits == "":
            return []
        if not probs_sequence:
            raise ValueError("probs_sequence must be non-empty for a non-empty payload")
        symbols: list[int] = []
        for probs in probs_sequence:
            symbols.append(stream.step(probs))
            if stream.finished:
                return symbols
        raise ValueError("probs_sequence exhausted before the payload could be fully encoded")

    def decode(
        self,
        symbols: list[int],
        probs_sequence: list[list[float] | np.ndarray],
        num_bits: int,
    ) -> str:
        if num_bits < 0:
            raise ValueError("num_bits must be non-negative")
        if num_bits == 0:
            return ""
        if len(symbols) > len(probs_sequence):
            raise ValueError("not enough distributions to decode the symbol sequence")
        stream = self.start_decode(num_bits=num_bits)
        for symbol, probs in zip(symbols, probs_sequence):
            stream.step(symbol, probs)
        return stream.finish(num_bits)

    def encode_symbol(self, bit: int, probs: list[float] | np.ndarray) -> int:
        symbols = self.encode(str(int(bool(bit))), [probs] * (self.precision + 8))
        return symbols[0]
