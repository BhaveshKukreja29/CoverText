from pathlib import Path

import torch

from covertext.common.model import load_model
from covertext.common.payload import generate_payload
from covertext.common.schema import StegoRecord, save_records
from covertext.detector.corpus import build_clean_corpus
from covertext.detector.ppl_detector import PPLDetector
from covertext.encoder.ac import ACEncoder
from covertext.encoder.mec_encoder import MECEncoder
from covertext.eval.metrics import (
    capacity_bpt,
    decode_accuracy_bit,
    decode_accuracy_exact,
    fluency_loss,
)

# ---------------- settings you can change ----------------
CONTEXT = "The Roman Empire reached its greatest territorial extent under"
NUM_BITS = 64          # secret payload size
SEED = 7               # payload seed
NATURAL_TOKENS = 60    # length of the plain model paragraph
CALIBRATION_N = 50     # clean paragraphs used to calibrate the detector
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# ----------------------------------------------------------

model, tokenizer = load_model(MODEL_ID)
payload = generate_payload(NUM_BITS, seed=SEED)

# 1. Natural paragraph: plain sampling with the same top_k / temperature as the encoders
torch.manual_seed(SEED)
inputs = tokenizer(CONTEXT, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(
        **inputs, do_sample=True, top_k=50, top_p=1.0, temperature=1.0,
        repetition_penalty=1.0, max_new_tokens=NATURAL_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
    )
natural = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# 2. Stego paragraphs: same context, same payload, two methods
methods = {"AC": ACEncoder(model, tokenizer), "MEC": MECEncoder(model, tokenizer)}
results = {}
for name, enc in methods.items():
    stego = enc.encode(payload, CONTEXT)
    recovered = enc.decode(stego, CONTEXT, NUM_BITS)
    results[name] = {
        "text": stego,
        "recovered": recovered,
        "tokens": len(tokenizer.encode(stego, add_special_tokens=False)),
    }

# 3. Detector, calibrated on clean human text (WikiText test split)
detector = PPLDetector(model, tokenizer, method="combined")
detector.calibrate(build_clean_corpus(tokenizer=tokenizer)[:CALIBRATION_N])

texts = {"NATURAL": natural, "AC": results["AC"]["text"], "MEC": results["MEC"]["text"]}
stats = {k: {"ppl": detector.compute_ppl(v), "kl": detector.compute_kl_divergence(v),
             "score": detector.score(v)} for k, v in texts.items()}

# 4. Report
lines = []


def log(text=""):
    print(text)
    lines.append(text)


log("=" * 72)
log(f"CONTEXT : {CONTEXT}")
log(f"PAYLOAD : {payload}  ({NUM_BITS} bits)")
log("=" * 72)
log("")
log("NATURAL (plain model sample, no hidden data)")
log(f"  {natural!r}")
for name in ("AC", "MEC"):
    r = results[name]
    log("")
    log(f"{name} STEGO")
    log(f"  {r['text']!r}")
    log(f"  recovered payload : {r['recovered']}")
    log(f"  exact match       : {decode_accuracy_exact(payload, r['recovered'])}   "
        f"bit accuracy: {decode_accuracy_bit(payload, r['recovered']):.3f}")

log("")
log("-" * 72)
log(f"{'':9s}{'tokens':>8s}{'bits/token':>12s}{'perplexity':>12s}{'KL':>8s}{'det. score':>12s}{'fluency loss':>14s}")
nat_ppl = stats["NATURAL"]["ppl"]
nat_tokens = len(tokenizer.encode(natural, add_special_tokens=False))
log(f"{'NATURAL':9s}{nat_tokens:8d}{'-':>12s}{nat_ppl:12.2f}{stats['NATURAL']['kl']:8.2f}"
    f"{stats['NATURAL']['score']:12.3f}{'-':>14s}")
for name in ("AC", "MEC"):
    s, r = stats[name], results[name]
    log(f"{name:9s}{r['tokens']:8d}{capacity_bpt(NUM_BITS, r['tokens']):12.2f}{s['ppl']:12.2f}"
        f"{s['kl']:8.2f}{s['score']:12.3f}{fluency_loss(s['ppl'], nat_ppl):14.2f}")
log("-" * 72)
log("bits/token : hidden bits per generated token (higher = more efficient)")
log("perplexity : lower = more fluent under the model")
log("det. score : higher = detector thinks it is more likely stego")
log("fluency loss: stego perplexity minus natural perplexity (positive = less fluent)")

Path("results").mkdir(exist_ok=True)
Path("results/simple_demo.txt").write_text("\n".join(lines), encoding="utf-8")
save_records(
    [StegoRecord(cover_text=CONTEXT, stego_text=results[n]["text"], payload_bits=payload,
                 method_name=n, model_name=MODEL_ID,
                 generation_settings={"top_k": 50, "temperature": 1.0})
     for n in ("AC", "MEC")],
    "results/simple_demo.json",
)
print("\nSaved results/simple_demo.txt and results/simple_demo.json")