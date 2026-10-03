import textwrap
from pathlib import Path

import torch

from covertext.common.model import load_model
from covertext.common.payload import generate_payload
from covertext.detector.corpus import build_clean_corpus
from covertext.detector.ppl_detector import PPLDetector
from covertext.encoder.ac import ACEncoder
from covertext.encoder.mec_encoder import MECEncoder
from covertext.eval.metrics import compute_auroc, decode_accuracy_bit, decode_accuracy_exact

# ---------------- settings ----------------
CONTEXT = (
    "The Roman Empire was one of the largest empires in ancient history. "
    "At its height it controlled territory across Europe, North Africa and the Middle East. "
    "Its expansion was driven by a powerful army, a network of roads and"
)
NUM_BITS = 200      # try 200 for a longer paragraph
MAX_TOKENS = 400
N_DETECT = 50       # samples per class in the detection test
TOP_K, TEMPERATURE = 30, 0.8
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# ------------------------------------------

model, tokenizer = load_model(MODEL_ID)
encoders = {
    "AC": ACEncoder(model, tokenizer, top_k=TOP_K, temperature=TEMPERATURE),
    "MEC": MECEncoder(model, tokenizer, top_k=TOP_K, temperature=TEMPERATURE),
}

lines = []


def log(text=""):
    print(text)
    lines.append(text)


def show(label, text):
    log(label)
    for ln in textwrap.wrap(" ".join(text.split()), 76):
        log("  " + ln)


def sample(n_tokens, seed):
    torch.manual_seed(seed)
    inputs = tokenizer(CONTEXT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, do_sample=True, top_k=TOP_K, top_p=1.0, temperature=TEMPERATURE,
            repetition_penalty=1.0, max_new_tokens=max(n_tokens, 1),
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def n_tok(text):
    return len(tokenizer.encode(text, add_special_tokens=False))


# ---------- Part 1: one long example ----------
payload = generate_payload(NUM_BITS, seed=7)
log("=" * 78)
log(f"PART 1: {NUM_BITS}-bit payload   {payload}")
log("=" * 78)
show("CONTEXT:", CONTEXT)
log()
show("NATURAL (no hidden data):", sample(60, 7))

for name, enc in encoders.items():
    stego = enc.encode(payload, CONTEXT, max_tokens=MAX_TOKENS)
    n = n_tok(stego)
    rec = enc.decode(stego, CONTEXT, NUM_BITS)
    log()
    show(f"{name} STEGO ({n} tokens, {NUM_BITS / n:.2f} bits/token):", stego)
    log(f"  recovered : {rec}")
    log(f"  exact match = {decode_accuracy_exact(payload, rec)}   "
        f"bit accuracy = {decode_accuracy_bit(payload, rec):.3f}")

# ---------- Part 2: can the detector tell stego from normal model text? ----------
log()
log("=" * 78)
log(f"PART 2: detection test ({N_DETECT} stego vs {N_DETECT} natural per method)")
log("=" * 78)
detector = PPLDetector(model, tokenizer, method="combined")
detector.calibrate(build_clean_corpus(tokenizer=tokenizer)[:50])

for name, enc in encoders.items():
    stego_scores, nat_scores = [], []
    for i in range(N_DETECT):
        p = generate_payload(NUM_BITS, seed=100 + i)
        stego = enc.encode(p, CONTEXT, max_tokens=MAX_TOKENS)
        natural = sample(n_tok(stego), 500 + i)
        stego_scores.append(detector.score(stego))
        nat_scores.append(detector.score(natural))
    auroc = compute_auroc(stego_scores + nat_scores, [1] * N_DETECT + [0] * N_DETECT)
    thr = sorted(nat_scores)[int(0.95 * (N_DETECT - 1))]  # ~5% false positives
    caught = sum(s > thr for s in stego_scores)
    log(f"{name}: mean score stego {sum(stego_scores) / N_DETECT:.3f} vs natural "
        f"{sum(nat_scores) / N_DETECT:.3f}   AUROC {auroc:.2f}   "
        f"stego flagged at 5% false-alarm threshold: {caught}/{N_DETECT}")

Path("results").mkdir(exist_ok=True)
Path("results/longer_demo.txt").write_text("\n".join(lines), encoding="utf-8")
print("\nSaved results/longer_demo.txt")