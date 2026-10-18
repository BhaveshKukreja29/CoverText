#!/usr/bin/env python3
"""Score PPL and CLS against AC and MEC. Writes AUROC per cell."""

from covertext.eval.benchmark import main_detectors

if __name__ == "__main__":
    main_detectors()
