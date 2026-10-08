"""Stub encoder — round trips payload as plain text for testing."""

from __future__ import annotations

from covertext.common.interfaces import Encoder


class StubEncoder(Encoder):
    """Embeds payload as literal text. For pipeline testing only."""

    @property
    def name(self) -> str:
        return "STUB"

    def encode(self, payload_bits: str, context: str, **kwargs) -> str:
        return f"{context} [STEGO:{payload_bits}]"

    def decode(self, stego_text: str, context: str, num_bits: int, **kwargs) -> str:
        marker = "[STEGO:"
        start = stego_text.index(marker) + len(marker)
        end = stego_text.index("]", start)
        bits = stego_text[start:end]
        return bits[:num_bits]
