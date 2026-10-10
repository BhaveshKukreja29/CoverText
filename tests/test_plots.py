"""Chart series are the experiment numbers, and the three files are PNGs."""

from covertext.eval.plots import (
    capacity_auroc_series,
    decode_accuracy_series,
    fluency_series,
    plot_summary,
)


def _row(encoder, detector, bits, capacity, accuracy, fluency, auroc):
    return {
        "encoder": encoder,
        "detector": detector,
        "payload_bits": bits,
        "capacity_bpt": capacity,
        "decode_accuracy_bit": accuracy,
        "fluency_loss": fluency,
        "auroc": auroc,
    }


def test_series_follow_payload_order_and_skip_undefined_fluency():
    summary = [
        _row("AC", "PPL", 8, 2.0, 1.0, 1.0, 0.6),
        _row("AC", "PPL", 16, 1.0, 1.0, None, 0.9),
        _row("AC", "CLS", 8, 2.0, 0.5, 3.0, 0.7),
    ]
    assert capacity_auroc_series(summary)["AC / PPL"] == [(2.0, 0.6), (1.0, 0.9)]
    assert capacity_auroc_series(summary)["AC / CLS"] == [(2.0, 0.7)]
    assert decode_accuracy_series(summary)["AC"] == [(8, 0.75), (16, 1.0)]
    assert fluency_series(summary)["AC"] == [(8, 2.0)]


def test_three_charts_render_from_summary(tmp_path):
    summary = [
        _row("STUB", "STUB", 8, 0.5, 1.0, None, 0.5),
        _row("STUB", "STUB", 16, 1.0, 1.0, None, 0.5),
    ]
    paths = plot_summary(summary, tmp_path)
    assert set(paths) == {"capacity_auroc", "decode_accuracy", "fluency_loss"}
    for path in paths.values():
        assert path.read_bytes().startswith(b"\x89PNG")
    assert fluency_series(summary) == {}
