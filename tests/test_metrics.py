import math

from covertext.eval.metrics import (
    capacity_bpt,
    compute_auroc,
    compute_perplexity,
    compute_roc_curve,
    decode_accuracy_bit,
    decode_accuracy_exact,
    fluency_loss,
)


def test_capacity_bpt():
    assert capacity_bpt(32, 16) == 2.0
    assert capacity_bpt(0, 10) == 0.0


def test_decode_accuracy_bit_perfect():
    assert decode_accuracy_bit("01101", "01101") == 1.0


def test_decode_accuracy_bit_partial():
    assert decode_accuracy_bit("01101", "01100") == 0.8


def test_decode_accuracy_bit_length_mismatch():
    acc = decode_accuracy_bit("0110", "01101")
    assert 0.0 <= acc <= 1.0
    assert acc == 4 / 5


def test_decode_accuracy_exact():
    assert decode_accuracy_exact("01101", "01101") is True
    assert decode_accuracy_exact("01101", "01100") is False


def test_fluency_loss():
    assert fluency_loss(25.0, 20.0) == 5.0


def test_compute_perplexity():
    assert abs(compute_perplexity([-1.0, -1.0, -1.0]) - math.e) < 0.01


def test_auroc_perfect_separation():
    scores = [0.1, 0.2, 0.9, 0.95]
    labels = [0, 0, 1, 1]
    assert compute_auroc(scores, labels) == 1.0


def test_auroc_random():
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [0, 1, 0, 1]
    assert abs(compute_auroc(scores, labels) - 0.5) < 1e-9


def test_compute_roc_curve_nonempty():
    fpr, tpr, thresholds = compute_roc_curve([0.1, 0.9], [0, 1])
    assert len(fpr) > 0
    assert len(tpr) > 0
    assert len(thresholds) > 0
