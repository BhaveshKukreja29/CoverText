# CoverText

Linguistic steganography and steganalysis in small language models. Final year project, Group 42.

Start here:

- [handoff.md](handoff.md) — what this project is, current status, read this first
- [plan.md](plan.md) — two phase, four person task breakdown
- [issues.md](issues.md) — concrete, self-assignable backlog (start with the Foundation issues)
- [research.md](research.md) / [final.md](final.md) — background, why this topic

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Layout

```
src/covertext/
  common/    shared schema + encoder/detector interfaces (FND 4)
  encoder/   AC, MEC, RC
  detector/  PPL, CLS, PROBE
  eval/      metrics, experiment runner, plotting
  demo/      CLI / web demo
tests/
data/        WikiText-103 cache, gitignored
results/     experiment output, gitignored
```

Nothing is implemented yet — this is just the scaffold. See issues.md, Foundation section, for what to build first.
