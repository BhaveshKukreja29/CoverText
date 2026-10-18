"""Stub sweep is repeatable, and the pins match this environment."""

import subprocess
import sys
from pathlib import Path

from covertext.common.pins import PINNED, pin_mismatches
from covertext.detector.stub import StubDetector
from covertext.encoder.stub import StubEncoder
from covertext.eval.plots import CAPACITY_AUROC, DECODE_ACCURACY, FLUENCY_LOSS
from covertext.eval.sweep import run_phase1_sweep


_CONTEXTS = [
    "The museum opened its north gallery in the spring.",
    "A river cuts through the valley below the ridge.",
    "Historians still argue about the treaty's wording.",
    "The telescope was repaired after the storm.",
]


def test_pins_match_the_installed_environment_and_pyproject():
    assert pin_mismatches() == []
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    for name, version in PINNED.items():
        assert f"{name}=={version}" in text


def test_stub_sweep_repeats_and_writes_the_three_charts(tmp_path):
    first = _sweep(tmp_path / "a")
    second = _sweep(tmp_path / "b")
    assert first["summary"] == second["summary"]
    assert len(first["summary"]) == 4
    assert {row["payload_bits"] for row in first["summary"]} == {8, 16, 32, 64}
    assert all(row["decode_accuracy_exact"] == 1.0 for row in first["summary"])
    for name in (CAPACITY_AUROC, DECODE_ACCURACY, FLUENCY_LOSS):
        chart = (tmp_path / "a" / name).read_bytes()
        assert chart.startswith(b"\x89PNG")


def test_reproduce_stub_command(tmp_path):
    output = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            "scripts/reproduce.py",
            "--stub",
            "--output",
            str(output),
            "--n-samples",
            "2",
            "--skip-version-check",
        ],
        check=True,
    )
    assert (output / "sweep.json").is_file()
    assert (output / CAPACITY_AUROC).read_bytes().startswith(b"\x89PNG")


def _sweep(path):
    return run_phase1_sweep(
        path,
        seed=0,
        n_samples=2,
        contexts=list(_CONTEXTS),
        clean_contexts=list(_CONTEXTS),
        encoders={"stub": StubEncoder()},
        detectors={"stub": StubDetector(seed=0)},
        model_name="none",
    )
