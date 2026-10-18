"""CLS head: pooling, cross-entropy, and held-out detection on a fake LM."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from covertext.common.interfaces import Detector
from covertext.detector.cls_detector import CLSDetector, last_token_pool, stego_probability
from covertext.eval.metrics import compute_auroc


def test_last_token_pool_uses_final_real_position():
    hidden = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [9.0, 9.0]],
            [[4.0, 0.0], [8.0, 8.0], [8.0, 8.0], [8.0, 8.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]])
    pooled = last_token_pool(hidden, mask)
    assert torch.equal(pooled[0], torch.tensor([3.0, 0.0]))
    assert torch.equal(pooled[1], torch.tensor([4.0, 0.0]))


def test_last_token_pool_rejects_empty_mask():
    hidden = torch.zeros(1, 2, 2)
    with pytest.raises(ValueError, match="empty"):
        last_token_pool(hidden, torch.zeros(1, 2))


def test_stego_probability_is_softmax_of_class_one():
    logits = torch.tensor([[0.3, -0.7], [2.0, 2.0]])
    got = stego_probability(logits)
    expected = torch.softmax(logits, dim=-1)[:, 1]
    assert torch.allclose(got, expected)
    gap = torch.sigmoid(logits[:, 1] - logits[:, 0])
    assert torch.allclose(got, gap)


def test_cross_entropy_matches_closed_form():
    uniform = F.cross_entropy(torch.tensor([[0.0, 0.0]]), torch.tensor([1]))
    assert abs(float(uniform) - math.log(2)) < 1e-6
    certain = F.cross_entropy(torch.tensor([[0.0, 40.0]]), torch.tensor([1]))
    assert float(certain) < 1e-6


def test_cross_entropy_gradient_matches_finite_difference():
    weight = torch.nn.Parameter(torch.tensor([[0.2, -0.4], [0.5, 0.1]], dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    features = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    target = torch.tensor([1])

    def loss_of(current_weight, current_bias):
        return F.cross_entropy(features @ current_weight.T + current_bias, target)

    loss_of(weight, bias).backward()
    eps = 1e-6
    numeric = torch.zeros_like(weight)
    with torch.no_grad():
        for i in range(2):
            for j in range(2):
                step = torch.zeros_like(weight)
                step[i, j] = eps
                plus = loss_of(weight + step, bias)
                minus = loss_of(weight - step, bias)
                numeric[i, j] = (plus - minus) / (2 * eps)
    assert torch.allclose(weight.grad, numeric, atol=1e-5)


class _SuffixTokenizer:
    """Token 1 marks stego, token 2 marks cover, token 9 is a prefix filler."""

    def __call__(self, texts, add_special_tokens=False, padding=False, truncation=False):
        rows = []
        for text in texts:
            label = 1 if "STEGO" in text else 2
            length = 6 if text.startswith("LONG") else 1
            row = [9] * (length - 1) + [label]
            rows.append(row)
        return {"input_ids": rows}


class _Body(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None):
        hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 2)
        hidden[:, :, 0] = (input_ids == 2).float()
        hidden[:, :, 1] = (input_ids == 1).float()
        pad = input_ids == 0
        hidden[:, :, 0] = torch.where(pad, torch.ones_like(hidden[:, :, 0]) * 5, hidden[:, :, 0])
        hidden[:, :, 1] = torch.where(pad, torch.ones_like(hidden[:, :, 1]) * 5, hidden[:, :, 1])
        return SimpleNamespace(last_hidden_state=hidden)


def _fake_model():
    return SimpleNamespace(model=_Body(), device=torch.device("cpu"), config=SimpleNamespace(hidden_size=2))


def _detector(**kwargs):
    defaults = dict(seed=0, lr=0.5, epochs=40, batch_size=8, weight_decay=0.0, max_length=4)
    defaults.update(kwargs)
    return CLSDetector(_fake_model(), _SuffixTokenizer(), **defaults)


def test_cls_implements_interface_and_refuses_unfitted_score():
    detector = _detector()
    assert isinstance(detector, Detector)
    assert detector.name == "CLS"
    with pytest.raises(ValueError, match="fit"):
        detector.score("clean")


def test_blank_text_is_rejected():
    detector = _detector()
    with pytest.raises(ValueError, match="non-empty"):
        detector.score("   ")
    with pytest.raises(ValueError, match="non-empty"):
        detector.sequence_features(["clean", " \n "])


def test_feature_chunks_match_a_single_forward():
    texts = ["clean", "LONG STEGO", "STEGO", "LONG clean"]
    wide = _detector(batch_size=8).sequence_features(texts)
    narrow = _detector(batch_size=1).sequence_features(texts)
    assert torch.equal(wide, narrow)


def test_batched_pool_matches_unpadded_suffix():
    detector = _detector(max_length=4)
    alone = detector.sequence_features(["clean"])
    batched = detector.sequence_features(["clean", "LONG clean"])
    assert torch.equal(alone[0], batched[0])
    assert torch.equal(alone[0], torch.tensor([1.0, 0.0]))
    long_feature = detector.sequence_features(["LONG STEGO"])
    assert torch.equal(long_feature[0], torch.tensor([0.0, 1.0]))


def test_suffix_truncation_keeps_the_stego_token():
    detector = _detector(max_length=2)
    feature = detector.sequence_features(["LONG STEGO"])
    assert torch.equal(feature[0], torch.tensor([0.0, 1.0]))


def test_fit_separates_held_out_text_and_matches_softmax():
    detector = _detector()
    train = [f"clean {i}" for i in range(16)] + [f"STEGO {i}" for i in range(16)]
    labels = [0] * 16 + [1] * 16
    stats = detector.fit(train, labels)
    assert stats["loss_after"] < stats["loss_before"]
    held_clean = [f"clean held {i}" for i in range(8)]
    held_stego = [f"STEGO held {i}" for i in range(8)]
    assert set(held_clean + held_stego).isdisjoint(train)
    scores = [detector.score(text) for text in held_clean + held_stego]
    auroc = compute_auroc(scores, [0] * 8 + [1] * 8)
    assert auroc == 1.0
    assert all(score < 0.1 for score in scores[:8])
    assert all(score > 0.9 for score in scores[8:])

    features = detector.sequence_features(["STEGO held 0"])
    standardized = (features - detector._mean) / detector._std
    with torch.no_grad():
        logits = detector.head(standardized)
        expected = float(torch.softmax(logits, dim=-1)[0, 1])
    assert abs(detector.score("STEGO held 0") - expected) < 1e-6


def test_save_load_preserves_probability(tmp_path):
    detector = _detector()
    texts = ["clean a", "clean b", "STEGO a", "STEGO b"]
    detector.fit(texts, [0, 0, 1, 1])
    path = tmp_path / "head.pt"
    detector.save(str(path))
    restored = _detector()
    restored.load(str(path))
    for text in ("clean z", "STEGO z"):
        assert abs(restored.score(text) - detector.score(text)) < 1e-6


def test_fit_requires_both_classes():
    detector = _detector()
    with pytest.raises(ValueError, match="both"):
        detector.fit(["clean", "clean"], [0, 0])


@pytest.mark.slow
def test_cls_auroc_on_real_model_against_stub_stego(model_fixture):
    from covertext.encoder.stub import StubEncoder

    model, tokenizer = model_fixture
    detector = CLSDetector(
        model, tokenizer, seed=0, epochs=12, lr=1e-2, batch_size=8, weight_decay=0.0
    )
    encoder = StubEncoder()
    train_clean = [f"Museum note number {i} about rivers and archives." for i in range(12)]
    test_clean = [f"Held out gallery note {i} about telescopes." for i in range(8)]
    train_stego = [encoder.encode(f"{i:08b}", train_clean[i]) for i in range(12)]
    test_stego = [encoder.encode(f"{i:08b}", test_clean[i]) for i in range(8)]
    assert set(train_clean + train_stego).isdisjoint(test_clean + test_stego)
    stats = detector.fit(train_clean + train_stego, [0] * 12 + [1] * 12)
    assert stats["loss_after"] < stats["loss_before"]
    scores = [detector.score(text) for text in test_clean + test_stego]
    auroc = compute_auroc(scores, [0] * 8 + [1] * 8)
    assert auroc > 0.9
