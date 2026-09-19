"""Minimum-entropy coupling encoder for linguistic steganography."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from covertext.common.interfaces import Encoder


def greedy_mec(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Greedy two-marginal minimum-entropy coupling (Kocaoglu et al., 2017)."""
    leftover_p = np.asarray(p, dtype=np.float64).copy()
    leftover_q = np.asarray(q, dtype=np.float64).copy()
    coupling = np.zeros((leftover_p.size, leftover_q.size), dtype=np.float64)
    while True:
        i = int(np.argmax(leftover_p))
        j = int(np.argmax(leftover_q))
        mass = min(leftover_p[i], leftover_q[j])
        if mass <= 1e-15:
            break
        coupling[i, j] += mass
        leftover_p[i] -= mass
        leftover_q[j] -= mass
    return coupling


class MECEncoder(Encoder):
    """Meteor-style encoder using minimum entropy coupling.

    Uses greedy MEC as an affinity, then a unique message-to-token matching so
    encode/decode is an exact inverse. Reference: Schroeder de Witt et al.,
    "Perfectly Secure Steganography Using Minimum Entropy Coupling", ICLR 2023.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        top_k: int = 50,
        temperature: float = 1.0,
        mec_algorithm: str = "fimec",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.temperature = temperature
        self.mec_algorithm = mec_algorithm

    @property
    def name(self) -> str:
        return "MEC"

    def encode(self, payload_bits: str, context: str, max_tokens: int = 100, **kwargs) -> str:
        max_tokens = kwargs.get("max_tokens", max_tokens)
        if payload_bits == "":
            return ""
        input_ids = self._tokenize(context)
        generated: list[int] = []
        remaining = payload_bits
        past = None
        for _ in range(max_tokens):
            if not remaining:
                break
            probs, token_ids, past = self._next_topk(input_ids, past)
            probs, token_ids = self._reversible(generated, probs, token_ids)
            bits_this_step = self._bits_per_step(len(token_ids), len(remaining))
            chunk = remaining[:bits_this_step]
            remaining = remaining[bits_this_step:]
            message = int(chunk, 2)
            token_index = self._message_to_token_index(probs, bits_this_step)[message]
            token_id = int(token_ids[token_index])
            generated.append(token_id)
            input_ids = torch.tensor([[token_id]], device=self.model.device)
        return self.tokenizer.decode(generated, skip_special_tokens=False)

    def decode(self, stego_text: str, context: str, num_bits: int, **kwargs) -> str:
        if num_bits == 0:
            return ""
        last_error: Exception | None = None
        for stego_ids in self._candidate_stego_ids(context, stego_text):
            try:
                return self._decode_ids(stego_ids, context, num_bits)
            except ValueError as exc:
                last_error = exc
        raise last_error if last_error else ValueError("could not recover stego tokens")

    def _decode_ids(self, stego_ids: list[int], context: str, num_bits: int) -> str:
        input_ids = self._tokenize(context)
        remaining = num_bits
        recovered: list[str] = []
        recovered_ids: list[int] = []
        past = None
        for token_id in stego_ids:
            if remaining <= 0:
                break
            probs, token_ids, past = self._next_topk(input_ids, past)
            probs, token_ids = self._reversible(recovered_ids, probs, token_ids)
            bits_this_step = self._bits_per_step(len(token_ids), remaining)
            token_to_message = {
                idx: msg
                for msg, idx in self._message_to_token_index(probs, bits_this_step).items()
            }
            try:
                position = token_ids.index(int(token_id))
            except ValueError as exc:
                raise ValueError("stego token is outside the encode-time top-k") from exc
            message = token_to_message[position]
            recovered.append(format(message, f"0{bits_this_step}b"))
            remaining -= bits_this_step
            recovered_ids.append(int(token_id))
            input_ids = torch.tensor([[int(token_id)]], device=self.model.device)
        bits = "".join(recovered)
        if len(bits) < num_bits:
            bits = bits.ljust(num_bits, "0")
        return bits[:num_bits]

    def _candidate_stego_ids(self, context: str, stego_text: str) -> list[list[int]]:
        del context
        encoded = self.tokenizer.encode(stego_text, add_special_tokens=False)
        return [list(encoded)] if encoded else []

    def _bits_per_step(self, vocab_k: int, remaining: int) -> int:
        if vocab_k < 2 or remaining <= 0:
            raise ValueError("cannot embed bits with a degenerate top-k distribution")
        return min(int(math.floor(math.log2(vocab_k))), remaining)

    def _message_to_token_index(self, probs: list[float], n_bits: int) -> dict[int, int]:
        n_messages = 1 << n_bits
        p = np.asarray(probs, dtype=np.float64)
        p = p / p.sum()
        q = np.full(n_messages, 1.0 / n_messages, dtype=np.float64)
        coupling = np.asarray(greedy_mec(p, q), dtype=np.float64)
        cost = -coupling.T
        message_ix, token_ix = linear_sum_assignment(cost)
        return {int(m): int(t) for m, t in zip(message_ix, token_ix)}

    def _reversible(
        self, generated: list[int], probs: list[float], token_ids: list[int]
    ) -> tuple[list[float], list[int]]:
        kept_probs: list[float] = []
        kept_ids: list[int] = []
        for prob, token_id in zip(probs, token_ids):
            trial = generated + [int(token_id)]
            text = self.tokenizer.decode(trial, skip_special_tokens=False)
            back = self.tokenizer.encode(text, add_special_tokens=False)
            if back == trial:
                kept_probs.append(prob)
                kept_ids.append(int(token_id))
        if len(kept_ids) < 2:
            return probs, token_ids
        total = sum(kept_probs)
        kept_probs = [p / total for p in kept_probs]
        return kept_probs, kept_ids

    def _tokenize(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return encoded["input_ids"].to(self.model.device)

    def _next_topk(self, input_ids: torch.Tensor, past_key_values=None):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = outputs.logits[0, -1, :].float() / self.temperature
        for attr in ("eos_token_id", "pad_token_id", "bos_token_id"):
            tid = getattr(self.tokenizer, attr, None)
            if tid is not None:
                logits[int(tid)] = -float("inf")
        k = min(self.top_k, int(torch.isfinite(logits).sum().item()))
        values, indices = torch.topk(logits, k, sorted=True)
        probs = torch.softmax(values, dim=-1)
        return probs.cpu().tolist(), indices.cpu().tolist(), outputs.past_key_values
