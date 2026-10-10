"""Encoder and detector benchmarks over the shared sample metrics."""

from covertext.common.interfaces import Detector
from covertext.common.payload import generate_payload
from covertext.common.schema import load_records
from covertext.encoder.stub import StubEncoder
from covertext.eval.benchmark import PAYLOAD_SIZES, run_detector_benchmark, run_encoder_benchmark
from covertext.eval.results import load_results, write_results


class NamedStub(StubEncoder):
    def __init__(self, label: str):
        self._label = label

    @property
    def name(self) -> str:
        return self._label


class MarkerDetector(Detector):
    @property
    def name(self) -> str:
        return "MARK"

    def score(self, text: str, **kwargs) -> float:
        return 1.0 if "[STEGO:" in text else 0.0


class FlatDetector(Detector):
    @property
    def name(self) -> str:
        return "FLAT"

    def score(self, text: str, **kwargs) -> float:
        return 0.5


def test_encoder_benchmark_covers_every_method_and_size(tmp_path):
    contexts = ["alpha beta", "gamma delta epsilon"]
    result = run_encoder_benchmark(
        {"ac": NamedStub("AC"), "mec": NamedStub("MEC")},
        contexts,
        n_per_size=1,
        seed=0,
        perplexity_fn=None,
        model_name="fake",
    )
    keys = {(row["encoder"], row["payload_bits"]) for row in result["summary"]}
    assert keys == {("AC", size) for size in PAYLOAD_SIZES} | {("MEC", size) for size in PAYLOAD_SIZES}
    assert all(row["decode_accuracy_exact"] == 1.0 for row in result["summary"])
    assert all(row["auroc"] is None for row in result["summary"])
    assert all(row["fluency_loss"] is None for row in result["summary"])
    path = tmp_path / "enc.json"
    write_results(path, result["summary"], result["records"])
    loaded = load_records(path)
    assert {record.method_name for record in loaded} == {"AC", "MEC"}
    assert len(loaded) == 2 * len(PAYLOAD_SIZES)


def test_encoder_benchmark_fluency_and_capacity_by_hand():
    contexts = ["alpha beta", "gamma delta epsilon"]
    result = run_encoder_benchmark(
        {"stub": NamedStub("STUB")},
        contexts,
        payload_sizes=[8],
        n_per_size=2,
        seed=0,
        perplexity_fn=len,
        model_name="fake",
    )
    expected = []
    for index in range(2):
        payload = generate_payload(8, seed=index)
        context = contexts[index]
        stego = f"{context} [STEGO:{payload}]"
        expected.append((8 / len(stego.split()), len(stego) - len(context), payload, stego))
    records = result["records"]
    for record, (capacity, fluency, payload, stego) in zip(records, expected):
        assert record.payload_bits == payload
        assert record.stego_text == stego
        assert record.generation_settings["capacity_bpt"] == capacity
        assert record.generation_settings["fluency_loss"] == fluency
    row = result["summary"][0]
    assert row["capacity_bpt"] == (expected[0][0] + expected[1][0]) / 2
    assert row["fluency_loss"] == (expected[0][1] + expected[1][1]) / 2
    assert row["decode_accuracy_bit"] == 1.0


def test_detector_benchmark_logs_auroc_for_every_cell(tmp_path):
    contexts = ["one two", "three four", "five six", "seven eight"]
    result = run_detector_benchmark(
        {"ac": NamedStub("AC"), "mec": NamedStub("MEC")},
        {"mark": MarkerDetector(), "flat": FlatDetector()},
        contexts,
        payload_sizes=list(PAYLOAD_SIZES),
        n_samples=2,
        seed=0,
        perplexity_fn=len,
        model_name="fake",
    )
    cells = {(row["encoder"], row["detector"], row["payload_bits"]) for row in result["summary"]}
    expected = {
        (encoder, detector, size)
        for encoder in ("AC", "MEC")
        for detector in ("MARK", "FLAT")
        for size in PAYLOAD_SIZES
    }
    assert cells == expected
    for row in result["summary"]:
        assert row["decode_accuracy_exact"] == 1.0
        if row["detector"] == "MARK":
            assert row["auroc"] == 1.0
        else:
            assert row["auroc"] == 0.5
        assert row["fluency_loss"] is not None
    path = tmp_path / "det.json"
    write_results(path, result["summary"], result["records"])
    data = load_results(path)
    assert len(data["summary"]) == 16
    assert all("auroc" in record["generation_settings"] for record in data["records"])
