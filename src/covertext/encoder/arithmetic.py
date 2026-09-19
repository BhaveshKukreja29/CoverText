"""Nayuki-style 32-bit arithmetic coder with streaming renormalization.

``DecodeStream`` (encoder) emits bits as the interval collapses. ``EncodeStream``
(decoder) reads payload bits into a code register and inverts the same updates.
Frequencies are scaled to ``FREQ_TOTAL`` (2^16) so they fit under the 32-bit
range. ``start_decode(num_bits)`` requires an explicit bit length; decoding
without it cannot recover the original payload length.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

PRECISION = 32
FREQ_TOTAL = 1 << 16
_FULL = 1 << PRECISION
_HALF = _FULL >> 1
_QUARTER = _HALF >> 1
_THREE_QUARTER = _QUARTER * 3
_MASK = _FULL - 1


def _interval_bounds(probs: Sequence[float]) -> np.ndarray:
    """Return integer CDF bounds of length ``len(probs)+1`` summing to ``FREQ_TOTAL``."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("probs must be a non-empty 1-d sequence")
    if np.any(p < 0) or np.any(~np.isfinite(p)):
        raise ValueError("probs must be finite and non-negative")
    total = p.sum()
    if total <= 0:
        raise ValueError("probs must sum to a positive value")
    p = p / total
    counts = np.maximum(np.floor(p * FREQ_TOTAL).astype(np.int64), 1)
    extra = int(counts.sum() - FREQ_TOTAL)
    if extra > 0:
        order = np.argsort(-counts)
        i = 0
        while extra > 0:
            idx = int(order[i % len(counts)])
            if counts[idx] > 1:
                counts[idx] -= 1
                extra -= 1
            i += 1
            if i > len(counts) * (extra + 2):
                break
    elif extra < 0:
        counts[int(np.argmax(p))] -= extra
    bounds = np.zeros(len(counts) + 1, dtype=np.int64)
    bounds[1:] = np.cumsum(counts)
    bounds[-1] = FREQ_TOTAL
    return bounds


class _RangeState:
    def __init__(self) -> None:
        self.low = 0
        self.high = _MASK

    def span(self) -> int:
        return self.high - self.low + 1

    def narrow(self, symbol: int, bounds: np.ndarray) -> None:
        rng = self.span()
        total = int(bounds[-1])
        self.high = self.low + rng * int(bounds[symbol + 1]) // total - 1
        self.low = self.low + rng * int(bounds[symbol]) // total


class DecodeStream:
    """Arithmetic encoder: consume symbols, emit bits."""

    def __init__(self, coder: ArithmeticCoder, num_bits: int) -> None:
        if num_bits < 0:
            raise ValueError("num_bits must be non-negative")
        self._coder = coder
        self.num_bits = num_bits
        self._state = _RangeState()
        self._pending = 0
        self.out: list[str] = []

    def _emit_bit(self, bit: int) -> None:
        self.out.append(str(bit))
        follow = 1 - bit
        self.out.extend(str(follow) for _ in range(self._pending))
        self._pending = 0

    def _renorm(self) -> None:
        low, high = self._state.low, self._state.high
        while True:
            if high < _HALF:
                self._emit_bit(0)
            elif low >= _HALF:
                self._emit_bit(1)
                low -= _HALF
                high -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTER:
                self._pending += 1
                low -= _QUARTER
                high -= _QUARTER
            else:
                break
            low = (low << 1) & _MASK
            high = ((high << 1) & _MASK) | 1
        self._state.low, self._state.high = low, high

    def step(self, symbol: int, probs: Sequence[float]) -> None:
        bounds = _interval_bounds(probs)
        if symbol < 0 or symbol >= len(probs):
            raise ValueError("symbol out of range")
        self._state.narrow(symbol, bounds)
        self._renorm()

    def clone(self) -> DecodeStream:
        other = DecodeStream(self._coder, self.num_bits)
        other._state.low = self._state.low
        other._state.high = self._state.high
        other._pending = self._pending
        other.out = list(self.out)
        return other

    def finish(self, num_bits: int | None = None) -> str:
        n = self.num_bits if num_bits is None else num_bits
        if n < 0:
            raise ValueError("num_bits must be non-negative")
        if len(self.out) >= n:
            return "".join(self.out[:n])
        self._pending += 1
        self._emit_bit((self._state.low >> (PRECISION - 2)) & 1)
        bits = "".join(self.out)
        if len(bits) >= n:
            return bits[:n]
        return bits.ljust(n, "0")


class EncodeStream:
    """Arithmetic decoder: consume payload bits, emit symbols."""

    def __init__(self, coder: ArithmeticCoder, bits: str) -> None:
        if any(c not in "01" for c in bits):
            raise ValueError("bits must contain only '0' and '1'")
        self._coder = coder
        self._state = _RangeState()
        self._bits = bits
        self._index = 0
        self._code = 0
        self.finished = bits == ""
        self._encoder = DecodeStream(coder, num_bits=len(bits))
        for _ in range(PRECISION):
            self._code = ((self._code << 1) & _MASK) | self._next_bit()

    def _next_bit(self) -> int:
        if self._index < len(self._bits):
            bit = int(self._bits[self._index])
            self._index += 1
            return bit
        return 0

    def _renorm(self) -> None:
        low, high, code = self._state.low, self._state.high, self._code
        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF
                high -= _HALF
                code -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTER:
                low -= _QUARTER
                high -= _QUARTER
                code -= _QUARTER
            else:
                break
            low = (low << 1) & _MASK
            high = ((high << 1) & _MASK) | 1
            code = ((code << 1) & _MASK) | self._next_bit()
        self._state.low, self._state.high, self._code = low, high, code

    def step(self, probs: Sequence[float]) -> int:
        bounds = _interval_bounds(probs)
        total = int(bounds[-1])
        rng = self._state.span()
        offset = self._code - self._state.low
        value = ((offset + 1) * total - 1) // rng
        symbol = int(np.searchsorted(bounds, value, side="right") - 1)
        symbol = max(0, min(symbol, len(probs) - 1))
        self._state.narrow(symbol, bounds)
        self._renorm()
        self._encoder.step(symbol, probs)
        recovered = self._encoder.clone().finish(len(self._bits))
        if recovered == self._bits:
            self.finished = True
        return symbol


class ArithmeticCoder:
    def __init__(self, precision: int = PRECISION):
        if precision != PRECISION:
            raise ValueError(f"only {PRECISION}-bit streaming precision is supported")
        self.precision = precision

    def encode(self, bits: str, distributions: Sequence[Sequence[float]]) -> list[int]:
        if bits == "":
            return []
        stream = self.start_encode(bits)
        symbols: list[int] = []
        for dist in distributions:
            symbols.append(stream.step(dist))
            if stream.finished:
                break
        return symbols

    def decode(
        self,
        symbols: Sequence[int],
        distributions: Sequence[Sequence[float]],
        num_bits: int,
    ) -> str:
        if num_bits == 0:
            return ""
        stream = self.start_decode(num_bits)
        for symbol, dist in zip(symbols, distributions):
            stream.step(symbol, dist)
        return stream.finish(num_bits)

    def start_encode(self, bits: str) -> EncodeStream:
        return EncodeStream(self, bits)

    def start_decode(self, num_bits: int) -> DecodeStream:
        return DecodeStream(self, num_bits)
