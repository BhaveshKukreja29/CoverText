#!/usr/bin/env python3
"""Reproduce the phase-one sweep and its three charts.

    python scripts/reproduce.py

The run is seeded, forced onto CPU in float32, and refused if the pinned
dependency versions are not the ones this tree was measured with. ``--stub``
runs the same writer against the stub encoder and detector so the command can
be checked without loading the language model.
"""

from __future__ import annotations

import argparse

from covertext.common.pins import assert_pinned_environment
from covertext.common.seed import seed_everything
from covertext.detector.stub import StubDetector
from covertext.encoder.stub import StubEncoder
from covertext.eval.benchmark import PAYLOAD_SIZES
from covertext.eval.sweep import run_phase1_sweep


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reproduce the phase-one sweep and charts")
    parser.add_argument("--output", default="results/phase1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--n-contexts", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stub", action="store_true", help="Use the stub encoder and detector")
    parser.add_argument("--skip-version-check", action="store_true")
    args = parser.parse_args(argv)
    if not args.skip_version_check:
        assert_pinned_environment()
    seed_everything(args.seed)
    if args.stub:
        contexts = [
            "The museum opened its north gallery in the spring.",
            "A river cuts through the valley below the ridge.",
            "Historians still argue about the treaty's wording.",
            "The telescope was repaired after the storm.",
        ]
        run_phase1_sweep(
            args.output,
            seed=args.seed,
            n_samples=args.n_samples,
            payload_sizes=PAYLOAD_SIZES,
            contexts=contexts,
            clean_contexts=list(contexts),
            encoders={"stub": StubEncoder()},
            detectors={"stub": StubDetector(seed=args.seed)},
            model_name="none",
            device=args.device,
        )
    else:
        run_phase1_sweep(
            args.output,
            seed=args.seed,
            n_samples=args.n_samples,
            n_contexts=args.n_contexts,
            device=args.device,
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
