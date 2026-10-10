"""CLI encode, decode, and detect against the stub implementations."""

import json

import pytest

from covertext.common.payload import text_to_bits
from covertext.demo.cli import decode_message, detect_text, encode_message, main
from covertext.detector.stub import StubDetector
from covertext.encoder.stub import StubEncoder


def test_message_round_trip_through_stub():
    encoder = StubEncoder()
    context = "cover sentence"
    encoded = encode_message(encoder, "hi", context)
    assert encoded["payload_bits"] == text_to_bits("hi")
    decoded = decode_message(encoder, encoded["stego_text"], context, encoded["num_bits"])
    assert decoded["payload_bits"] == encoded["payload_bits"]
    assert decoded["message"] == "hi"


def test_cli_encode_decode_detect(capsys):
    context = "cover sentence"
    main(["encode", "--message", "hi", "--context", context, "--encoder", "stub"])
    encoded = json.loads(capsys.readouterr().out)
    main(
        [
            "decode",
            "--text",
            encoded["stego_text"],
            "--context",
            context,
            "--num-bits",
            str(encoded["num_bits"]),
            "--encoder",
            "stub",
        ]
    )
    decoded = json.loads(capsys.readouterr().out)
    assert decoded["message"] == "hi"
    main(["detect", "--text", encoded["stego_text"], "--detectors", "stub", "--seed", "0"])
    scores = json.loads(capsys.readouterr().out)["scores"]
    assert set(scores) == {"STUB"}
    assert scores["STUB"] == StubDetector(seed=0).score(encoded["stego_text"])


def test_detect_scores_each_detector():
    class Other(StubDetector):
        @property
        def name(self) -> str:
            return "OTHER"

        def score(self, text, **kwargs):
            return 0.25

    result = detect_text("plain", [StubDetector(seed=1), Other()])
    assert set(result["scores"]) == {"STUB", "OTHER"}
    assert result["scores"]["OTHER"] == 0.25
    assert 0.0 <= result["scores"]["STUB"] <= 1.0


def test_cli_requires_one_payload_source():
    with pytest.raises(SystemExit) as caught:
        main(["encode", "--encoder", "stub"])
    assert caught.value.code == 2


def test_raw_payload_flag(capsys):
    main(["encode", "--payload", "0110", "--context", "ctx", "--encoder", "stub"])
    encoded = json.loads(capsys.readouterr().out)
    assert encoded["payload_bits"] == "0110"
    assert encoded["stego_text"] == "ctx [STEGO:0110]"
