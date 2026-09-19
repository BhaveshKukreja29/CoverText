import math

import pytest

from covertext.common.interfaces import Detector
from covertext.eval.metrics import compute_auroc, compute_roc_curve
from covertext.encoder.stub import StubEncoder
from covertext.detector.ppl_detector import PPLDetector


def test_ppl_detector_implements_interface():
    dummy_model = type("M", (), {"device": "cpu"})()
    detector = PPLDetector(dummy_model, tokenizer=object())
    assert isinstance(detector, Detector)


def test_score_requires_calibration():
    detector = PPLDetector(type("M", (), {"device": "cpu"})(), tokenizer=object())
    with pytest.raises(ValueError, match="calibrated"):
        detector.score("some text")


@pytest.mark.slow
def test_ppl_detector_returns_score(model_fixture):
    model, tokenizer = model_fixture
    detector = PPLDetector(model, tokenizer)
    detector.calibrate(["The sky is blue.", "History is written by people."])
    score = detector.score("some text about history and science")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@pytest.mark.slow
def test_compute_ppl_positive(model_fixture):
    model, tokenizer = model_fixture
    detector = PPLDetector(model, tokenizer)
    assert detector.compute_ppl("The quick brown fox") > 0


@pytest.mark.slow
def test_compute_kl_positive(model_fixture):
    model, tokenizer = model_fixture
    detector = PPLDetector(model, tokenizer)
    text = "The quick brown fox"
    kl = detector.compute_kl_divergence(text)
    ppl = detector.compute_ppl(text)
    assert math.isfinite(kl)
    assert abs(kl - math.log(ppl)) > 1e-6


@pytest.mark.slow
def test_calibrate(model_fixture):
    model, tokenizer = model_fixture
    detector = PPLDetector(model, tokenizer)
    detector.calibrate(["The sky is blue.", "History is written by people."])
    assert detector.reference_ppl is not None
    assert detector.reference_kl is not None


@pytest.mark.slow
def test_roc_against_stub_stego(model_fixture):
    model, tokenizer = model_fixture
    detector = PPLDetector(model, tokenizer)
    encoder = StubEncoder()
    clean = [f"This is clean sample number {i} about museums and rivers." for i in range(100)]
    stego = [encoder.encode(f"{i:08b}", clean[i]) for i in range(100)]
    detector.calibrate(clean[:20])
    scores = [detector.score(t) for t in clean + stego]
    labels = [0] * 100 + [1] * 100
    auroc = compute_auroc(scores, labels)
    assert 0.0 <= auroc <= 1.0


@pytest.mark.slow
def test_roc_curve_renders(model_fixture, tmp_path):
    model, tokenizer = model_fixture
    detector = PPLDetector(model, tokenizer)
    encoder = StubEncoder()
    clean = [f"Clean paragraph {i} with enough words to score." for i in range(20)]
    stego = [encoder.encode("1010", clean[i]) for i in range(20)]
    detector.calibrate(clean[:10])
    scores = [detector.score(t) for t in clean + stego]
    labels = [0] * 20 + [1] * 20
    fpr, tpr, thresholds = compute_roc_curve(scores, labels)
    assert len(fpr) > 0
    assert len(tpr) > 0
    assert len(thresholds) > 0
    del tmp_path
