"""Shared next-token utilities used by model loading and both encoders."""

from __future__ import annotations

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def mask_special_logits(logits: torch.Tensor, tokenizer: PreTrainedTokenizerFast) -> torch.Tensor:
    for attr in ("eos_token_id", "pad_token_id", "bos_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            logits[int(tid)] = -float("inf")
    return logits


def topk_probs(
    logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    finite = int(torch.isfinite(logits).sum().item())
    k = min(top_k, max(finite, 1), logits.size(-1))
    values, indices = torch.topk(logits, k, sorted=True)
    return torch.softmax(values, dim=-1), indices


def next_token_topk(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerFast,
    temperature: float = 1.0,
    top_k: int = 50,
    past_key_values=None,
) -> tuple[list[float], list[int], object]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
    logits = outputs.logits[0, -1, :].float() / temperature
    logits = mask_special_logits(logits, tokenizer)
    probs, indices = topk_probs(logits, top_k)
    return probs.cpu().tolist(), indices.cpu().tolist(), outputs.past_key_values


def reversible_subset(
    tokenizer: PreTrainedTokenizerFast,
    generated: list[int],
    probs: list[float],
    token_ids: list[int],
) -> tuple[list[float], list[int]]:
    """Keep only tokens that round-trip through decode → encode."""
    kept_probs: list[float] = []
    kept_ids: list[int] = []
    for prob, token_id in zip(probs, token_ids):
        trial = generated + [int(token_id)]
        text = tokenizer.decode(trial, skip_special_tokens=False)
        back = tokenizer.encode(text, add_special_tokens=False)
        if back == trial:
            kept_probs.append(prob)
            kept_ids.append(int(token_id))
    if not kept_ids:
        raise ValueError("no reversible tokens in top-k")
    if len(kept_ids) == 1:
        return [1.0], kept_ids
    total = sum(kept_probs)
    return [p / total for p in kept_probs], kept_ids


def stego_token_ids(tokenizer: PreTrainedTokenizerFast, stego_text: str) -> list[int]:
    return tokenizer.encode(stego_text, add_special_tokens=False)
