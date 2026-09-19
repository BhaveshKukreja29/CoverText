"""Minimum-entropy coupling encoder for linguistic steganography."""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from covertext.common.generation import next_token_topk, reversible_subset
from covertext.common.interfaces import Encoder


def greedy_mec(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Greedy two-marginal minimum-entropy coupling (Kocaoglu et al., 2017)."""
    leftover_p = np.asarray(p, dtype=np.float64).copy()
    leftover_q = np.asarray(q, dtype=np.float64).copy()
    leftover_p = leftover_p / leftover_p.sum()
    leftover_q = leftover_q / leftover_q.sum()
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


def _inverse_cdf(probs: np.ndarray, u: float) -> int:
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0
    idx = int(np.searchsorted(cdf, min(max(u, 0.0), 1.0 - 1e-12), side="right"))
    return min(idx, len(probs) - 1)


def _column(coupling: np.ndarray, message: int) -> np.ndarray | None:
    col = coupling[:, message]
    total = float(col.sum())
    if total <= 1e-15:
        return None
    return col / total


class MECEncoder(Encoder):
    """Meteor-style encoder using minimum entropy coupling.

    Tokens are sampled from the coupling conditional ``P(X | M = m)`` with a
    shared PRNG so decode can invert the same draw. The induced token
    marginal matches the language-model top-k distribution. Reference:
    Schroeder de Witt et al., ICLR 2023.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        top_k: int = 50,
        temperature: float = 1.0,
        mec_algorithm: str = "greedy",
        seed: int = 0,
    ):
        if mec_algorithm not in {"greedy", "fimec"}:
            raise ValueError("mec_algorithm must be 'greedy' or 'fimec'")
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.temperature = temperature
        self.mec_algorithm = mec_algorithm
        self.seed = seed

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
        step = 0
        for _ in range(max_tokens):
            if not remaining:
                break
            probs, token_ids, past = next_token_topk(
                self.model,
                input_ids,
                self.tokenizer,
                temperature=self.temperature,
                top_k=self.top_k,
                past_key_values=past,
            )
            probs, token_ids = reversible_subset(
                self.tokenizer, generated, probs, token_ids
            )
            bits_this_step = self._bits_per_step(probs, len(remaining))
            message = int(remaining[:bits_this_step], 2) if bits_this_step else 0
            token_index = self._sample_token_index(probs, message, bits_this_step, step)
            remaining = remaining[bits_this_step:]
            token_id = int(token_ids[token_index])
            generated.append(token_id)
            input_ids = torch.tensor([[token_id]], device=self.model.device)
            step += 1
        if remaining:
            raise RuntimeError("Payload truncated: max_tokens reached")
        return self.tokenizer.decode(generated, skip_special_tokens=False)

    def decode(self, stego_text: str, context: str, num_bits: int, **kwargs) -> str:
        if num_bits == 0:
            return ""
        stego_ids = self.tokenizer.encode(stego_text, add_special_tokens=False)
        if not stego_ids:
            raise ValueError("could not recover stego tokens")
        return self._decode_ids(stego_ids, context, num_bits)

    def _decode_ids(self, stego_ids: list[int], context: str, num_bits: int) -> str:
        input_ids = self._tokenize(context)
        remaining = num_bits
        recovered: list[str] = []
        recovered_ids: list[int] = []
        past = None
        step = 0
        for token_id in stego_ids:
            if remaining <= 0:
                break
            probs, token_ids, past = next_token_topk(
                self.model,
                input_ids,
                self.tokenizer,
                temperature=self.temperature,
                top_k=self.top_k,
                past_key_values=past,
            )
            probs, token_ids = reversible_subset(
                self.tokenizer, recovered_ids, probs, token_ids
            )
            bits_this_step = self._bits_per_step(probs, remaining)
            try:
                position = token_ids.index(int(token_id))
            except ValueError as exc:
                raise ValueError("stego token is outside the encode-time top-k") from exc
            if bits_this_step:
                message = self._recover_message(probs, position, bits_this_step)
                recovered.append(format(message, f"0{bits_this_step}b"))
                remaining -= bits_this_step
            recovered_ids.append(int(token_id))
            input_ids = torch.tensor([[int(token_id)]], device=self.model.device)
            step += 1
        bits = "".join(recovered)
        if remaining > 0 or len(bits) < num_bits:
            raise RuntimeError("Payload truncated: max_tokens reached")
        return bits[:num_bits]

    def _bits_per_step(self, probs: list[float], remaining: int) -> int:
        """Largest n such that each token couples to at most one message.

        Greedy MEC assigns a token to a second message only when p(x) > 2^{-n}.
        Capping n at floor(log2(1/max p)) keeps P(M|X) a delta, so decode is
        unique, while the token marginal remains the LM distribution.
        """
        if remaining <= 0 or len(probs) < 2:
            return 0
        max_p = max(probs)
        if max_p <= 0 or max_p >= 1:
            return 0
        cap_mass = int(math.floor(math.log2(1.0 / max_p)))
        cap_k = int(math.floor(math.log2(len(probs))))
        return min(cap_k, cap_mass, remaining)

    def _coupling(self, probs: list[float], n_bits: int) -> np.ndarray:
        n_messages = 1 << n_bits
        p = np.asarray(probs, dtype=np.float64)
        p = p / p.sum()
        q = np.full(n_messages, 1.0 / n_messages, dtype=np.float64)
        return greedy_mec(p, q)

    def _rng(self, step: int) -> random.Random:
        return random.Random(f"{self.seed}:{step}")

    def _sample_token_index(
        self, probs: list[float], message: int, n_bits: int, step: int
    ) -> int:
        u = self._rng(step).random()
        if n_bits == 0:
            p = np.asarray(probs, dtype=np.float64)
            return _inverse_cdf(p / p.sum(), u)
        col = _column(self._coupling(probs, n_bits), message)
        if col is None:
            raise ValueError("message has no mass under the coupling")
        return _inverse_cdf(col, u)

    def _recover_message(self, probs: list[float], token: int, n_bits: int) -> int:
        row = self._coupling(probs, n_bits)[token]
        if float(row.sum()) <= 0:
            raise ValueError("token has no mass under the coupling")
        return int(np.argmax(row))

    def _tokenize(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return encoded["input_ids"].to(self.model.device)
