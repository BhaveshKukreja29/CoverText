"""Stub detector — returns a random score for testing."""

from __future__ import annotations

import random

from covertext.common.interfaces import Detector


class StubDetector(Detector):
    """Returns a random score. For pipeline testing only."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "STUB"

    def score(self, text: str, **kwargs) -> float:
        return self._rng.random()
