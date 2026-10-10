"""Direct dependency pins. The reproduce command refuses a different install."""

from __future__ import annotations

import importlib.metadata

# Names are the importlib.metadata distribution names.
PINNED: dict[str, str] = {
    "torch": "2.14.0",
    "transformers": "5.17.0",
    "datasets": "5.0.1",
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "scikit-learn": "1.7.2",
    "matplotlib": "3.10.9",
    "accelerate": "1.15.0",
    "tqdm": "4.70.1",
    "pytest": "9.1.1",
}


def pin_mismatches() -> list[str]:
    """Return ``name==installed, expected version`` lines for every mismatch."""
    found: list[str] = []
    for name, expected in PINNED.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found.append(f"{name} is not installed, expected {expected}")
            continue
        if installed != expected:
            found.append(f"{name}=={installed}, expected {expected}")
    return found


def assert_pinned_environment() -> None:
    mismatches = pin_mismatches()
    if mismatches:
        raise RuntimeError("dependency pins differ:\n" + "\n".join(mismatches))
