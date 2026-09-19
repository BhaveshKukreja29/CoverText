"""Arithmetic-coding encoder for linguistic steganography."""

from __future__ import annotations

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from covertext.common.generation import isolate_stego_context, next_token_topk, reversible_subset
from covertext.common.interfaces import Encoder
from covertext.encoder.arithmetic import ArithmeticCoder


class ACEncoder(Encoder):
    """Hide a payload in LLM text by steering tokens with arithmetic coding.

    Reference: Ziegler et al., "Neural Linguistic Steganography", EMNLP 2019.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        precision: int = 32,
        top_k: int = 50,
        temperature: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.coder = ArithmeticCoder(precision=precision)
        self.top_k = top_k
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "AC"

    def encode(self, payload_bits: str, context: str, max_tokens: int = 100, **kwargs) -> str:
        max_tokens = kwargs.get("max_tokens", max_tokens)
        if payload_bits == "":
            return ""
        enc = self.coder.start_encode(payload_bits)
        input_ids = self._tokenize(context)
        generated: list[int] = []
        past = None
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
                token_id = int(token_ids[enc.step(probs)])
            generated.append(token_id)
            input_ids = torch.tensor([[token_id]], device=self.model.device)
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
        past = None
        generated: list[int] = []
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
                self.tokenizer, generated, probs, token_ids
            )
            try:
                index = token_ids.index(int(token_id))
            except ValueError as exc:
                raise ValueError(
                    "stego token is outside the encode-time top-k; cannot decode"
                ) from exc
            if len(token_ids) >= 2:
                stream.step(index, probs)
            generated.append(int(token_id))
            input_ids = torch.tensor([[int(token_id)]], device=self.model.device)
        return stream.finish(num_bits)

    def _tokenize(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(
            isolate_stego_context(text), return_tensors="pt", add_special_tokens=False
        )
        return encoded["input_ids"].to(self.model.device)
