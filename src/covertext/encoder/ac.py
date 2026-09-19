"""Arithmetic-coding encoder for linguistic steganography."""

from __future__ import annotations

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

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
            probs, token_ids, past = self._next_topk(input_ids, past_key_values=past)
            index = enc.step(probs)
            token_id = int(token_ids[index])
            generated.append(token_id)
            input_ids = torch.tensor([[token_id]], device=self.model.device)
            if enc.finished:
                break
        return self.tokenizer.decode(generated, skip_special_tokens=False)

    def decode(self, stego_text: str, context: str, num_bits: int, **kwargs) -> str:
        if num_bits == 0:
            return ""
        context_ids = self._tokenize(context)
        full_ids = self._tokenize(context + stego_text)
        stego_ids = full_ids[0, context_ids.shape[1] :].tolist()
        if not stego_ids:
            stego_ids = self.tokenizer.encode(stego_text, add_special_tokens=False)
        stream = self.coder.start_decode(num_bits=num_bits)
        input_ids = context_ids
        past = None
        for token_id in stego_ids:
            probs, token_ids, past = self._next_topk(input_ids, past_key_values=past)
            try:
                index = token_ids.index(int(token_id))
            except ValueError as exc:
                raise ValueError(
                    "stego token is outside the encode-time top-k; cannot decode"
                ) from exc
            stream.step(index, probs)
            input_ids = torch.tensor([[int(token_id)]], device=self.model.device)
        return stream.finish(num_bits)

    def _tokenize(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return encoded["input_ids"].to(self.model.device)

    def _next_topk(
        self, input_ids: torch.Tensor, past_key_values=None
    ) -> tuple[list[float], list[int], object]:
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
