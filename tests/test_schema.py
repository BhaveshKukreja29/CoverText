import json

from covertext.common.interfaces import Detector, Encoder
from covertext.common.schema import StegoRecord, load_records, save_records
from covertext.detector.stub import StubDetector
from covertext.encoder.stub import StubEncoder


def test_stego_record_roundtrip():
    record = StegoRecord(
        cover_text="cover",
        stego_text="stego",
        payload_bits="01101",
        method_name="STUB",
        model_name="test-model",
        generation_settings={"temperature": 1.0},
    )
    restored = StegoRecord.from_json(record.to_json())
    assert restored == record


def test_load_records_accepts_experiment_document(tmp_path):
    record = StegoRecord(
        cover_text="a",
        stego_text="b",
        payload_bits="01",
        method_name="STUB",
        model_name="m",
        record_id="id-1",
    )
    path = tmp_path / "wrapped.json"
    path.write_text(
        json.dumps({"summary": [{"auroc": 0.5}], "records": [record.to_dict()]}),
        encoding="utf-8",
    )
    assert load_records(path) == [record]


def test_save_load_records(tmp_path):
    records = [
        StegoRecord(
            cover_text="a",
            stego_text="b",
            payload_bits="01",
            method_name="STUB",
            model_name="m",
            record_id="id-1",
        ),
        StegoRecord(
            cover_text="c",
            stego_text="d",
            payload_bits="10",
            method_name="STUB",
            model_name="m",
            record_id="id-2",
        ),
    ]
    path = tmp_path / "records.json"
    save_records(records, path)
    loaded = load_records(path)
    assert loaded == records


def test_stub_encoder_roundtrip():
    encoder = StubEncoder()
    payload = "01101011"
    stego = encoder.encode(payload, "hello")
    recovered = encoder.decode(stego, "hello", len(payload))
    assert recovered == payload


def test_stub_detector_returns_score():
    detector = StubDetector(seed=0)
    score = detector.score("some text")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_stub_encoder_implements_interface():
    assert isinstance(StubEncoder(), Encoder)


def test_stub_detector_implements_interface():
    assert isinstance(StubDetector(), Detector)
