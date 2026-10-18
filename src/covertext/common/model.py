"""Model loading utility for Qwen2.5-0.5B-Instruct."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from covertext.common.generation import mask_special_logits, topk_probs

MODEL_ID: str = "Qwen/Qwen2.5-0.5B-Instruct"


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(
    model_id: str = MODEL_ID, device: str | None = None
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Load a causal LM and its tokenizer, moved to ``device`` in eval mode."""
    device = _resolve_device(device)
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, tokenizer


def get_next_token_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    context: str,
    top_k: int | None = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return next-token probabilities and vocab ids for ``context``."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    encoded = tokenizer(context, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
    logits = outputs.logits[0, -1, :].float() / temperature
    logits = mask_special_logits(logits, tokenizer)
    if top_k is None:
        probs = torch.softmax(logits, dim=-1)
        token_ids = torch.arange(probs.size(0), device=probs.device)
        return probs, token_ids
    probs, indices = topk_probs(logits, top_k)
    full = torch.zeros(logits.size(0), dtype=probs.dtype, device=probs.device)
    full[indices] = probs
    token_ids = torch.arange(full.size(0), device=full.device)
    return full, token_ids


def generate_completion(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    max_new_tokens: int = 50,
) -> str:
    """Generate a completion for ``prompt``, excluding the prompt text."""
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = generated[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
