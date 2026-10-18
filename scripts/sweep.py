#!/usr/bin/env python3
"""Run the phase-one encoder/detector matrix and write the charts."""

import argparse

from covertext.common.pins import assert_pinned_environment
from covertext.common.seed import seed_everything
from covertext.eval.sweep import run_phase1_sweep


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase-one AC/MEC against PPL/CLS")
    parser.add_argument("--output", default="results/phase1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--n-contexts", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    assert_pinned_environment()
    seed_everything(args.seed)
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
