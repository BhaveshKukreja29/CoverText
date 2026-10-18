"""Construct encoders and detectors by name. Callers share this table."""

from __future__ import annotations

import inspect

from covertext.common.interfaces import Detector, Encoder
from covertext.detector.cls_detector import CLSDetector
from covertext.detector.ppl_detector import PPLDetector
from covertext.detector.stub import StubDetector
from covertext.encoder.ac import ACEncoder
from covertext.encoder.mec_encoder import MECEncoder
from covertext.encoder.stub import StubEncoder


def build_encoder(name: str, model=None, tokenizer=None, **kwargs) -> Encoder:
    """``stub`` needs no model. ``ac`` and ``mec`` require one."""
    key = name.strip().lower()
    if key == "stub":
        return StubEncoder()
    cls = {"ac": ACEncoder, "mec": MECEncoder}.get(key)
    if cls is None:
        raise ValueError(f"unknown encoder '{name}'. Expected stub, ac, or mec")
    if model is None or tokenizer is None:
        raise ValueError(f"encoder '{name}' requires a loaded model and tokenizer")
    return cls(model, tokenizer, **_accepted(cls, kwargs))


def build_detector(name: str, model=None, tokenizer=None, **kwargs) -> Detector:
    """``stub`` needs no model. ``ppl`` and ``cls`` require one."""
    key = name.strip().lower()
    if key == "stub":
        return StubDetector(seed=int(kwargs.get("seed", 42)))
    cls = {"ppl": PPLDetector, "cls": CLSDetector}.get(key)
    if cls is None:
        raise ValueError(f"unknown detector '{name}'. Expected stub, ppl, or cls")
    if model is None or tokenizer is None:
        raise ValueError(f"detector '{name}' requires a loaded model and tokenizer")
    return cls(model, tokenizer, **_accepted(cls, kwargs))


def _accepted(cls: type, kwargs: dict) -> dict:
    params = inspect.signature(cls.__init__).parameters
    allowed = {name for name in params if name not in {"self", "model", "tokenizer"}}
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise ValueError(f"unknown settings for {cls.__name__}: {unknown}")
    return {key: kwargs[key] for key in kwargs if key in allowed}
