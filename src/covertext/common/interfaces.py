"""Abstract interfaces for encoders and detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Encoder(ABC):
    """Abstract encoder interface. All encoders (AC, MEC, STUB) implement this."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'AC', 'MEC', 'STUB'."""

    @abstractmethod
    def encode(self, payload_bits: str, context: str, **kwargs) -> str:
        """Encode a bitstring into stego text.

        Args:
            payload_bits: Binary string, e.g. ``"01101011"``.
            context: Cover text / prompt to condition generation on.
            **kwargs: Method-specific settings (temperature, top_k, max_tokens, etc.).

        Returns:
            The generated text carrying the hidden payload.
        """

    @abstractmethod
    def decode(self, stego_text: str, context: str, num_bits: int, **kwargs) -> str:
        """Decode (recover) a bitstring from stego text.

        Args:
            stego_text: The steganographic text to decode.
            context: The same context used during encoding.
            num_bits: Number of payload bits to recover.
            **kwargs: Must match the settings used during encode.

        Returns:
            Recovered binary string.
        """


class Detector(ABC):
    """Abstract detector interface. All detectors (PPL, CLS, STUB) implement this."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'PPL', 'CLS', 'STUB'."""

    @abstractmethod
    def score(self, text: str, **kwargs) -> float:
        """Score a piece of text for likelihood of carrying a hidden payload.

        Args:
            text: The text to analyze.
            **kwargs: Method-specific settings.

        Returns:
            A float in [0, 1] where higher means more likely stego.
        """
