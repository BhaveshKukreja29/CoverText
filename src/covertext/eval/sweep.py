"""Phase-one matrix: both encoders, both detectors, four payload sizes, then charts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from covertext.common.interfaces import Detector, Encoder
from covertext.common.model import MODEL_ID
from covertext.common.seed import seed_everything
from covertext.eval.benchmark import PAYLOAD_SIZES, run_detector_benchmark
from covertext.eval.plots import plot_summary
from covertext.eval.results import write_results
from covertext.eval.samples import count_tokens


def run_phase1_sweep(
    output_dir: str | Path,
    *,
    seed: int = 0,
    n_samples: int = 4,
    payload_sizes: tuple[int, ...] | list[int] = PAYLOAD_SIZES,
    contexts: list[str] | None = None,
    clean_contexts: list[str] | None = None,
    encoders: dict[str, Encoder] | None = None,
    detectors: dict[str, Detector] | None = None,
    tokenizer=None,
    model_name: str = "none",
    device: str = "cpu",
    n_contexts: int = 8,
) -> dict[str, Any]:
    """Run the detector matrix and write ``sweep.json`` plus the three charts.

    Pass ``encoders`` and ``detectors`` to run the same path on stubs. Otherwise
    the Qwen encoders and detectors are built on ``device``.
    """
    if (encoders is None) != (detectors is None):
        raise ValueError("pass both encoders and detectors, or neither")
    seed_everything(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    token_counter = None
    perplexity = None
    if encoders is None or detectors is None:
        from covertext.common.model import load_model
        from covertext.common.registry import build_detector, build_encoder
        from covertext.detector.corpus import build_clean_corpus, get_encoder_contexts
        from covertext.detector.ppl_detector import PPLDetector

        model, tokenizer = load_model(device=device)
        encoders = {
            "ac": build_encoder("ac", model, tokenizer),
            "mec": build_encoder("mec", model, tokenizer, seed=seed),
        }
        detectors = {
            "ppl": build_detector("ppl", model, tokenizer),
            "cls": build_detector("cls", model, tokenizer, seed=seed),
        }
        contexts = get_encoder_contexts(tokenizer=tokenizer)[:n_contexts]
        clean_contexts = build_clean_corpus(tokenizer=tokenizer)[:n_contexts]
        model_name = MODEL_ID
        perplexity = PPLDetector(model, tokenizer).compute_ppl

        def token_counter(text: str, _tokenizer=tokenizer) -> int:
            return count_tokens(text, _tokenizer)

    if contexts is None or clean_contexts is None:
        raise ValueError("contexts and clean_contexts are required when encoders are injected")
    result = run_detector_benchmark(
        encoders,
        detectors,
        contexts,
        payload_sizes=list(payload_sizes),
        n_samples=n_samples,
        seed=seed,
        token_counter=token_counter,
        perplexity_fn=perplexity,
        model_name=model_name,
        clean_contexts=clean_contexts,
    )
    results_path = output / "sweep.json"
    write_results(results_path, result["summary"], result["records"])
    charts = plot_summary(result["summary"], output)
    manifest = {
        "seed": seed,
        "n_samples": n_samples,
        "n_contexts": len(contexts),
        "payload_sizes": list(payload_sizes),
        "device": device,
        "model_name": model_name,
        "results": str(results_path),
        "charts": {name: str(path) for name, path in charts.items()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"summary": result["summary"], "records": result["records"], "manifest": manifest, "charts": charts}
