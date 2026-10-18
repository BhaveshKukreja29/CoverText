"""Seed the generators the sweep actually uses.

Payloads, splits, the stub detector, and the classifier head each construct
their own ``Random`` or ``torch.Generator`` from the sweep seed. This function
also seeds the process-wide generators so any other draw follows the same seed.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
