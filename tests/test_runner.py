"""Experiment runner: held-out split, hand-checked metrics, results file."""

import json

import pytest

from covertext.common.interfaces import Detector
from covertext.common.payload import generate_payload
from covertext.common.schema import load_records
from covertext.encoder.stub import StubEncoder
from covertext.eval.results import load_results
from covertext.eval.runner import (
    BUILTIN_CONTEXTS,
    ExperimentConfig,
    config_from_mapping,
    main,
    run_experiment,
)
from covertext.eval.samples import split_train_eval


class MarkerDetector(Detector):
    @property
    def name(self) -> str:
        return "MARK"

    def score(self, text: str, **kwargs) -> float:
        return 1.0 if "[STEGO:" in text else 0.0


class RememberDetector(Detector):
    def __init__(self):
        self.fit_texts: list[str] = []
        self.calibrated: list[str] = []

    @property
    def name(self) -> str:
        return "REMEMBER"

    def fit(self, texts, labels):
        self.fit_texts = list(texts)

    def calibrate(self, texts):
        self.calibrated = list(texts)

    def score(self, text: str, **kwargs) -> float:
        return 0.5


class CalibrateOnly(Detector):
    def __init__(self):
        self.calibrated: list[str] = []

    @property
    def name(self) -> str:
        return "CAL"

    def calibrate(self, texts):
        self.calibrated = list(texts)

    def score(self, text: str, **kwargs) -> float:
        return 1.0 if "[STEGO:" in text else 0.0


def test_split_is_deterministic_and_disjoint():
    items = [f"paragraph {i}" for i in range(7)]
    first = split_train_eval(items, 0.5, seed=3)
    assert split_train_eval(items, 0.5, seed=3) == first
    train, held = first
    assert set(train).isdisjoint(held)
    assert set(train) | set(held) == set(items)
    assert train and held


def test_runner_metrics_match_hand_computation(tmp_path):
    contexts = ["one two", "three four five", "six seven", "eight nine ten"]
    config = ExperimentConfig(
        encoder_name="stub",
        detector_name="mark",
        payload_sizes=[8],
        n_samples=2,
        seed=0,
        model_name="none",
        output_path=str(tmp_path / "run.json"),
        contexts=contexts,
    )
    result = run_experiment(
        config,
        StubEncoder(),
        MarkerDetector(),
        perplexity_fn=len,
    )
    _, held = split_train_eval(contexts, 0.5, seed=0)
    expected = []
    for index in range(2):
        payload = generate_payload(8, seed=500_000 + index)
        context = held[index % len(held)]
        stego = f"{context} [STEGO:{payload}]"
        expected.append(
            {
                "payload": payload,
                "context": context,
                "stego": stego,
                "capacity": 8 / len(stego.split()),
                "fluency": len(stego) - len(context),
            }
        )
    records = result["records"]
    assert [record.payload_bits for record in records] == [item["payload"] for item in expected]
    assert [record.cover_text for record in records] == [item["context"] for item in expected]
    assert [record.stego_text for record in records] == [item["stego"] for item in expected]
    for record, item in zip(records, expected):
        settings = record.generation_settings
        assert settings["capacity_bpt"] == item["capacity"]
        assert settings["fluency_loss"] == item["fluency"]
        assert settings["decode_accuracy_bit"] == 1.0
        assert settings["decode_accuracy_exact"] is True
        assert settings["auroc"] == 1.0
    row = result["summary"][0]
    assert row["auroc"] == 1.0
    assert row["decode_accuracy_exact"] == 1.0
    assert row["capacity_bpt"] == sum(item["capacity"] for item in expected) / 2
    assert row["fluency_loss"] == sum(item["fluency"] for item in expected) / 2
    assert row["encoder"] == "STUB"
    assert row["detector"] == "MARK"

    loaded = load_records(tmp_path / "run.json")
    assert [record.payload_bits for record in loaded] == [item["payload"] for item in expected]
    summary = load_results(tmp_path / "run.json")["summary"]
    assert summary[0]["auroc"] == 1.0


def test_fit_and_calibration_never_see_eval_text():
    contexts = ["one two", "three four five", "six seven", "eight nine ten"]
    config = ExperimentConfig(
        encoder_name="stub",
        detector_name="remember",
        payload_sizes=[8],
        n_samples=2,
        seed=0,
        contexts=contexts,
    )
    train, held = split_train_eval(contexts, 0.5, seed=0)
    fitted = RememberDetector()
    fit_result = run_experiment(config, StubEncoder(), fitted)
    n = config.n_samples
    fit_covers = fitted.fit_texts[:n]
    assert set(fit_covers) <= set(train)
    assert set(fit_covers).isdisjoint(held)
    eval_stego = {record.stego_text for record in fit_result["records"]}
    assert eval_stego.isdisjoint(fitted.fit_texts)
    assert fitted.calibrated == []

    calibrated = CalibrateOnly()
    run_experiment(config, StubEncoder(), calibrated)
    assert set(calibrated.calibrated) <= set(train)
    assert set(calibrated.calibrated).isdisjoint(held)


def test_decode_failure_is_a_total_miss():
    class Boom(StubEncoder):
        @property
        def name(self) -> str:
            return "BOOM"

        def decode(self, stego_text, context, num_bits, **kwargs):
            raise RuntimeError("nope")

    config = ExperimentConfig(
        encoder_name="boom",
        detector_name="mark",
        payload_sizes=[8],
        n_samples=1,
        seed=1,
        contexts=["alpha beta", "gamma delta"],
    )
    result = run_experiment(config, Boom(), MarkerDetector())
    settings = result["records"][0].generation_settings
    assert settings["decode_accuracy_bit"] == 0.0
    assert settings["decode_accuracy_exact"] is False
    assert "nope" in settings["decode_error"]


def test_separate_clean_corpus_is_the_negative_class():
    class PromptLooksStego(Detector):
        def __init__(self):
            self.calibrated: list[str] = []

        @property
        def name(self) -> str:
            return "PROMPT"

        def calibrate(self, texts):
            self.calibrated = list(texts)

        def score(self, text: str, **kwargs) -> float:
            if "[STEGO:" in text:
                return 1.0
            if text.startswith("clean"):
                return 0.0
            return 1.0

    prompts = ["prompt one", "prompt two", "prompt three", "prompt four"]
    clean = ["clean one", "clean two", "clean three", "clean four"]
    config = ExperimentConfig(
        encoder_name="stub",
        detector_name="prompt",
        payload_sizes=[8],
        n_samples=2,
        seed=0,
        contexts=prompts,
        clean_contexts=clean,
    )
    detector = PromptLooksStego()
    result = run_experiment(config, StubEncoder(), detector)
    assert result["summary"][0]["auroc"] == 1.0
    assert all(text.startswith("clean") for text in detector.calibrated)
    assert set(detector.calibrated).isdisjoint(prompts)


def test_config_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown"):
        config_from_mapping(
            {"encoder_name": "stub", "detector_name": "stub", "payload_sizes": [8], "nope": 1}
        )


def test_runner_script_writes_stub_results(tmp_path):
    path = tmp_path / "stub.json"
    main(
        [
            "--encoder",
            "stub",
            "--detector",
            "stub",
            "--payload-sizes",
            "8,16",
            "--n-samples",
            "2",
            "--output",
            str(path),
            "--contexts",
            str(_write_contexts(tmp_path)),
        ]
    )
    data = load_results(path)
    assert [row["payload_bits"] for row in data["summary"]] == [8, 16]
    assert all(row["decode_accuracy_exact"] == 1.0 for row in data["summary"])
    assert all(0.0 <= row["auroc"] <= 1.0 for row in data["summary"])
    assert len(BUILTIN_CONTEXTS) >= 2


def _write_contexts(tmp_path):
    path = tmp_path / "contexts.json"
    path.write_text(json.dumps(["alpha beta", "gamma delta", "epsilon zeta"]), encoding="utf-8")
    return path
