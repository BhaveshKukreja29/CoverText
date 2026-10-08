"""Minimum-entropy coupling encoder for linguistic steganography."""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from covertext.common.generation import (
    isolate_stego_context,
    next_token_topk,
    reversible_subset,
)
from covertext.common.interfaces import Encoder
from covertext.encoder.arithmetic import ArithmeticCoder


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


def bin_partition(probs: list[float], n_bits: int) -> tuple[list[int], list[float]]:
    """Assign each token to exactly one of ``2**n_bits`` bins, balancing mass.

    Tokens are never split. Bin masses generally will not equal ``2^{-n}`` on
    continuous softmaxes; arithmetic coding consumes those unequal masses.
    """
    n_bins = 1 << n_bits
    p = np.asarray(probs, dtype=np.float64)
    if p.size < n_bins:
        raise ValueError("need at least one token per bin")
    p = p / p.sum()
    assignment = np.empty(p.size, dtype=np.int64)
    masses = np.zeros(n_bins, dtype=np.float64)
    for index in np.argsort(-p, kind="stable"):
        bin_id = int(np.argmin(masses))
        assignment[index] = bin_id
        masses[bin_id] += p[index]
    return assignment.tolist(), masses.tolist()


class MECEncoder(Encoder):
    """Meteor-style encoder using disjoint bin coupling plus arithmetic coding.

    Real softmax masses never partition into exact ``2^{-n}`` subsets, so a
    split-free coupling against a uniform message prior embeds nothing. Each
    token is assigned to one bin; the bin is chosen with ``ArithmeticCoder``
    under the bin masses, then a token is sampled inside that bin. Decode reads
    the bin from the observed token. The token marginal remains ``p(x)``.
    Reference: Schroeder de Witt et al., ICLR 2023; Phase-one construction.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        top_k: int = 50,
        temperature: float = 1.0,
        mec_algorithm: str = "greedy",
        seed: int = 0,
        precision: int = 32,
    ):
        if mec_algorithm != "greedy":
            raise NotImplementedError(
                f"mec_algorithm '{mec_algorithm}' not supported; use 'greedy'"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.temperature = temperature
        self.mec_algorithm = mec_algorithm
        self.seed = seed
        self.coder = ArithmeticCoder(precision=precision)

    @property
    def name(self) -> str:
        return "MEC"

    def encode(self, payload_bits: str, context: str, max_tokens: int = 100, **kwargs) -> str:
        max_tokens = kwargs.get("max_tokens", max_tokens)
        if payload_bits == "":
            return ""
        enc = self.coder.start_encode(payload_bits)
        input_ids = self._tokenize(context)
        generated: list[int] = []
        past = None
        step = 0
        for _ in range(max_tokens):
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
            if len(token_ids) < 2:
                token_id = int(token_ids[0])
            else:
                n_bits = self._bin_bits(probs)
                assignment, masses = bin_partition(probs, n_bits)
                bin_index = enc.step(masses)
                token_index = self._sample_in_bin(probs, assignment, bin_index, step)
                token_id = int(token_ids[token_index])
            generated.append(token_id)
            input_ids = torch.tensor([[token_id]], device=self.model.device)
            step += 1
            if enc.finished:
                break
        if not enc.finished:
            raise RuntimeError("Payload truncated: max_tokens reached")
        return self.tokenizer.decode(generated, skip_special_tokens=False)

    def decode(self, stego_text: str, context: str, num_bits: int, **kwargs) -> str:
        if num_bits == 0:
            return ""
        stego_ids = self.tokenizer.encode(stego_text, add_special_tokens=False)
        if not stego_ids:
            raise ValueError("could not recover stego tokens")
        stream = self.coder.start_decode(num_bits)
        input_ids = self._tokenize(context)
        recovered_ids: list[int] = []
        past = None
        for token_id in stego_ids:
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
            try:
                position = token_ids.index(int(token_id))
            except ValueError as exc:
                raise ValueError("stego token is outside the encode-time top-k") from exc
            if len(token_ids) >= 2:
                n_bits = self._bin_bits(probs)
                assignment, masses = bin_partition(probs, n_bits)
                stream.step(int(assignment[position]), masses)
            recovered_ids.append(int(token_id))
            input_ids = torch.tensor([[int(token_id)]], device=self.model.device)
        return stream.finish(num_bits)

    def _bin_bits(self, probs: list[float]) -> int:
        return max(1, int(math.floor(math.log2(len(probs)))))

    def _rng(self, step: int) -> random.Random:
        return random.Random(f"{self.seed}:{step}")

    def _sample_in_bin(
        self, probs: list[float], assignment: list[int], bin_index: int, step: int
    ) -> int:
        p = np.asarray(probs, dtype=np.float64)
        members = np.flatnonzero(np.asarray(assignment) == bin_index)
        if members.size == 0:
            raise ValueError("arithmetic coder selected an empty bin")
        cond = p[members]
        cond = cond / cond.sum()
        local = _inverse_cdf(cond, self._rng(step).random())
        return int(members[local])

    def _tokenize(self, text: str) -> torch.Tensor:
        text = isolate_stego_context(text)
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return encoded["input_ids"].to(self.model.device)
