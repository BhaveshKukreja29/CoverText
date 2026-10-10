"""The smoke path recovers the message, then returns a detector score."""

import pytest

from covertext.common.payload import text_to_bits
from covertext.demo.smoke import main, run_smoke


def test_smoke_stub_recovers_the_message(capsys):
    result = run_smoke("stub", "hi", "cover sentence", seed=1)
    assert result["decoded"]["payload_bits"] == text_to_bits("hi")
    assert result["decoded"]["message"] == "hi"
    assert set(result["detected"]["scores"]) == {"STUB"}
    score = result["detected"]["scores"]["STUB"]
    assert 0.0 <= score <= 1.0
    main(["--encoder", "stub", "--message", "hi", "--context", "cover sentence", "--seed", "1"])
    printed = capsys.readouterr().out
    assert "ok encoder=STUB" in printed
    assert text_to_bits("hi") in printed


@pytest.mark.slow
@pytest.mark.parametrize("encoder_name", ["ac", "mec"])
def test_smoke_real_encoder_recovers_the_message(encoder_name):
    result = run_smoke(encoder_name, "hi", "The museum opened its north gallery in the spring.")
    assert result["decoded"]["payload_bits"] == result["encoded"]["payload_bits"]
    assert result["decoded"]["message"] == "hi"
    assert set(result["detected"]["scores"]) == {"STUB", "PPL"}
