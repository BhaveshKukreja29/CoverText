import statistics
import time
from pathlib import Path

import torch

from covertext.common.model import load_model
from covertext.common.payload import generate_payload
from covertext.encoder.mec_encoder import MECEncoder

model, tokenizer = load_model()
encoder = MECEncoder(model, tokenizer)

CONTEXT = "The history of science is"
WRONG_CONTEXT = "Yesterday I went to the market and"
SIZES = [8, 16, 32, 64]
N = 10  # payloads per size

lines = []


def log(text=""):
    print(text)
    lines.append(text)


def natural_text(context, n_tokens, seed):
    torch.manual_seed(seed)
    inputs = tokenizer(context, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=max(n_tokens, 1),
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def safe_decode(text, context, size):
    try:
        return encoder.decode(text, context, size)
    except Exception:
        return None


def bit_agreement(a, b):
    if a is None or not a:
        return 0.5  # treat a failed decode as chance level
    n = min(len(a), len(b))
    return sum(x == y for x, y in zip(a, b)) / n


log(f"CONTEXT: {CONTEXT!r}   payloads per size: {N}")
log("")

for size in SIZES:
    rows = []
    for i in range(N):
        payload = generate_payload(size, seed=1000 + i)
        stego = encoder.encode(payload, CONTEXT)
        recovered = encoder.decode(stego, CONTEXT, size)
        n_tokens = len(tokenizer.encode(stego, add_special_tokens=False))
        natural = natural_text(CONTEXT, n_tokens, seed=5000 + i)

        wrong = safe_decode(stego, WRONG_CONTEXT, size)
        nat_dec = safe_decode(natural, CONTEXT, size)

        rows.append({
            "payload": payload,
            "stego": stego,
            "ok": recovered == payload,
            "tokens": n_tokens,
            "bpt": size / max(n_tokens, 1),
            "wrong_exact": wrong == payload,
            "wrong_agree": bit_agreement(wrong, payload),
            "nat_exact": nat_dec == payload,
            "nat_agree": bit_agreement(nat_dec, payload),
        })

    bpts = [r["bpt"] for r in rows]
    toks = [r["tokens"] for r in rows]
    pooled = size * N / sum(toks)
    best = max(rows, key=lambda r: r["bpt"])
    worst = min(rows, key=lambda r: r["bpt"])

    log(f"[{size}-bit payload]  ({N} trials)")
    log(f"  round trip ok        : {sum(r['ok'] for r in rows)}/{N}")
    log(f"  tokens used          : mean {statistics.mean(toks):.1f}, min {min(toks)}, max {max(toks)}")
    log(f"  bits/token           : mean {statistics.mean(bpts):.2f}, sd {statistics.pstdev(bpts):.2f}, "
        f"min {min(bpts):.2f}, max {max(bpts):.2f}, pooled {pooled:.2f}")
    log(f"  control: wrong context   -> exact matches {sum(r['wrong_exact'] for r in rows)}/{N}, "
        f"mean bit agreement {statistics.mean(r['wrong_agree'] for r in rows):.2f}")
    log(f"  control: natural text    -> exact matches {sum(r['nat_exact'] for r in rows)}/{N}, "
        f"mean bit agreement {statistics.mean(r['nat_agree'] for r in rows):.2f}")
    log(f"  highest bits/token ({best['bpt']:.2f}): {CONTEXT}{best['stego']!r}")
    log(f"  lowest bits/token  ({worst['bpt']:.2f}): {CONTEXT}{worst['stego']!r}")
    log("")

out = Path("outputs")
out.mkdir(exist_ok=True)
(out / "mec_stats.txt").write_text("\n".join(lines), encoding="utf-8")
print("Saved to outputs/mec_stats.txt")