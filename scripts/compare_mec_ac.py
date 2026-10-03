import statistics
import time
from pathlib import Path

import torch

from covertext.common.model import load_model
from covertext.common.payload import generate_payload
from covertext.common.schema import StegoRecord, save_records
from covertext.detector.corpus import build_clean_corpus, get_encoder_contexts
from covertext.detector.ppl_detector import PPLDetector
from covertext.encoder.ac import ACEncoder
from covertext.encoder.mec_encoder import MECEncoder
from covertext.eval.metrics import (
    capacity_bpt,
    compute_auroc,
    decode_accuracy_bit,
    decode_accuracy_exact,
)

SIZES = [16, 32, 64]
N = 20            # contexts per size
CAL_N = 100       # clean paragraphs used only for detector calibration
CONTEXT_TOKENS = 30
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

model, tokenizer = load_model(MODEL_ID)
encoders = {"MEC": MECEncoder(model, tokenizer), "AC": ACEncoder(model, tokenizer)}

# --- data: contexts come from the train split, clean text from the test split (disjoint) ---
contexts_full = get_encoder_contexts(tokenizer=tokenizer)
clean_corpus = build_clean_corpus(tokenizer=tokenizer)


def trunc(text, n):
    ids = tokenizer.encode(text, add_special_tokens=False)[:n]
    return tokenizer.decode(ids)


contexts = [trunc(p, CONTEXT_TOKENS) for p in contexts_full[:N]]
cal_texts = clean_corpus[:CAL_N]
eval_clean = clean_corpus[CAL_N:CAL_N + N]

detector = PPLDetector(model, tokenizer, method="combined")
detector.calibrate(cal_texts)


def natural_sample(context, n_tokens, seed):
    """Plain sampling from the model with the same top_k/temperature as the encoders.
    top_p and repetition_penalty are set explicitly because Qwen's generation config
    would otherwise change the distribution."""
    torch.manual_seed(seed)
    inputs = tokenizer(context, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, do_sample=True, top_k=50, top_p=1.0, temperature=1.0,
            repetition_penalty=1.0, max_new_tokens=max(n_tokens, 1),
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def safe_score(text):
    try:
        return detector.score(text)
    except Exception:
        return None


lines = []


def log(text=""):
    print(text)
    lines.append(text)


Path("results").mkdir(exist_ok=True)
log(f"model: {MODEL_ID}   contexts per size: {N}   context length: {CONTEXT_TOKENS} tokens")
log("AUROC: 0.5 = detector cannot tell the classes apart, 1.0 = perfectly separated")
log("")

for size in SIZES:
    for name, enc in encoders.items():
        rows, records = [], []
        for i, context in enumerate(contexts):
            payload = generate_payload(size, seed=1000 + i)
            try:
                t0 = time.time()
                stego = enc.encode(payload, context)
                dt = time.time() - t0
            except Exception as exc:
                rows.append({"failed": str(exc)})
                continue
            try:
                recovered = enc.decode(stego, context, size)
            except Exception:
                recovered = ""
            n_tok = len(tokenizer.encode(stego, add_special_tokens=False))
            natural = natural_sample(context, n_tok, seed=5000 + i)
            human = trunc(eval_clean[i], n_tok)
            rows.append({
                "exact": decode_accuracy_exact(payload, recovered),
                "bit": decode_accuracy_bit(payload, recovered),
                "tokens": n_tok,
                "bpt": capacity_bpt(size, n_tok),
                "time": dt,
                "s_stego": safe_score(stego),
                "s_natural": safe_score(natural),
                "s_human": safe_score(human),
                "example": (context, stego, natural, payload, recovered),
            })
            records.append(StegoRecord(
                cover_text=context, stego_text=stego, payload_bits=payload,
                method_name=name, model_name=MODEL_ID,
                generation_settings={"top_k": 50, "temperature": 1.0},
            ))

        ok = [r for r in rows if "failed" in r]
        good = [r for r in rows if "failed" not in r]
        log(f"[{size}-bit] {name}   encode failures: {len(ok)}/{len(rows)}")
        if not good:
            log("")
            continue
        bpts = [r["bpt"] for r in good]
        toks = [r["tokens"] for r in good]
        log(f"  exact recovery   : {sum(r['exact'] for r in good)}/{len(good)}   "
            f"mean bit accuracy {statistics.mean(r['bit'] for r in good):.3f}")
        log(f"  tokens           : mean {statistics.mean(toks):.1f}  (min {min(toks)}, max {max(toks)})")
        log(f"  bits/token       : mean {statistics.mean(bpts):.2f}  sd {statistics.pstdev(bpts):.2f}  "
            f"pooled {size * len(good) / sum(toks):.2f}")
        log(f"  encode time      : mean {statistics.mean(r['time'] for r in good):.2f}s")

        def auroc(neg_key):
            pairs = [(r["s_stego"], r[neg_key]) for r in good
                     if r["s_stego"] is not None and r[neg_key] is not None]
            if not pairs:
                return float("nan")
            scores = [p[0] for p in pairs] + [p[1] for p in pairs]
            labels = [1] * len(pairs) + [0] * len(pairs)
            return compute_auroc(scores, labels)

        log(f"  detector AUROC   : stego vs model-natural {auroc('s_natural'):.2f}   "
            f"stego vs human WikiText {auroc('s_human'):.2f}")
        ctx, stego, natural, payload, recovered = good[0]["example"]
        log(f"  example context  : {ctx!r}")
        log(f"  example stego    : {stego!r}")
        log(f"  example natural  : {natural!r}")
        log(f"  payload/recovered: {payload} / {recovered}")
        log("")
        save_records(records, f"results/records_{name}_{size}bit.json")

Path("results/compare_mec_ac.txt").write_text("\n".join(lines), encoding="utf-8")
print("Saved results/compare_mec_ac.txt and results/records_*.json")