"""Perplexity and KL divergence baseline detector."""

from __future__ import annotations

import math
import statistics

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from covertext.common.interfaces import Detector


_ENTROPY_CHUNK = 4096


def _chunked_entropy(logits: torch.Tensor, log_z: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of softmax(logits) without allocating (T, V) probabilities."""
    weighted_logits = torch.zeros(logits.size(0), dtype=logits.dtype, device=logits.device)
    vocab = logits.size(-1)
    for start in range(0, vocab, _ENTROPY_CHUNK):
        sl = logits[:, start : start + _ENTROPY_CHUNK]
        weighted_logits += (torch.exp(sl - log_z) * sl).sum(dim=-1)
    return log_z.squeeze(1) - weighted_logits


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class PPLDetector(Detector):
    """Score text by how much token-level statistics deviate from the base model.

    ``compute_ppl`` is sequence perplexity. ``compute_kl_divergence`` is the
    mean conditional entropy deficit ``-log P(x_t) - H(P_t)``. Natural text
    has expectation 0; stego steering yields a positive deficit. Scoring
    requires ``calibrate()`` first.

    ``score`` is a two-sided anomaly detector: deviations *below* the clean
    reference (unusually fluent text) and *above* it (typical stego inflation of
    PPL / entropy deficit) are treated the same. Stego usually increases both
    statistics; the absolute deviation is kept so calibration remains
    well-defined when a sample is atypically fluent.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        method: str = "combined",
        reference_ppl: float | None = None,
        reference_kl: float | None = None,
    ):
        if method not in {"ppl", "kl", "combined"}:
            raise ValueError("method must be 'ppl', 'kl', or 'combined'")
        self.model = model
        self.tokenizer = tokenizer
        self.method = method
        self.reference_ppl = reference_ppl
        self.reference_kl = reference_kl
        self._ppl_std: float | None = None
        self._kl_std: float | None = None

    @property
    def name(self) -> str:
        return "PPL"

    def _forward(self, text: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not text:
            return None
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(self.model.device)
        if input_ids.shape[1] < 2:
            return None
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits[0].float()
        return logits, input_ids

    def compute_token_log_probs(self, text: str) -> list[float]:
        packed = self._forward(text)
        if packed is None:
            return []
        logits, input_ids = packed
        token_ids = input_ids[0, 1:]
        logits_t = logits[:-1].float()
        log_z = torch.logsumexp(logits_t, dim=-1)
        chosen = logits_t.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        return (chosen - log_z).cpu().tolist()

    def compute_ppl(self, text: str) -> float:
        log_probs = self.compute_token_log_probs(text)
        if not log_probs:
            return float("inf")
        return math.exp(-statistics.mean(log_probs))

    def compute_kl_divergence(self, text: str) -> float:
        """Mean conditional entropy deficit: mean(-log P(x_t) - H(P_t)).

        Under the model policy this has expectation 0. Stego encoders push mass
        into lower-probability candidates, so the deficit is typically positive.
        Entropy is accumulated in vocabulary chunks so a 150k-class softmax is
        never fully materialized.
        """
        packed = self._forward(text)
        if packed is None:
            return 0.0
        logits, input_ids = packed
        token_ids = input_ids[0, 1:]
        logits_t = logits[:-1].float()
        log_z = torch.logsumexp(logits_t, dim=-1, keepdim=True)
        chosen = logits_t.gather(1, token_ids.unsqueeze(1))
        neg_log_p = (log_z - chosen).squeeze(1)
        entropy = _chunked_entropy(logits_t, log_z)
        return float((neg_log_p - entropy).mean().cpu())

    def calibrate(self, clean_texts: list[str]) -> None:
        ppls = [self.compute_ppl(t) for t in clean_texts if t]
        kls = [self.compute_kl_divergence(t) for t in clean_texts if t]
        finite_ppls = [p for p in ppls if math.isfinite(p)]
        if not finite_ppls or not kls:
            raise ValueError("calibration requires at least one non-empty text")
        self.reference_ppl = statistics.mean(finite_ppls)
        self.reference_kl = statistics.mean(kls)
        self._ppl_std = statistics.pstdev(finite_ppls) if len(finite_ppls) > 1 else None
        self._kl_std = statistics.pstdev(kls) if len(kls) > 1 else None

    def score(self, text: str, **kwargs) -> float:
        if self.reference_ppl is None or self.reference_kl is None:
            raise ValueError("PPLDetector must be calibrated before scoring")
        ppl_score = self._deviation_score(
            self.compute_ppl(text), self.reference_ppl, self._ppl_std
        )
        kl_score = self._deviation_score(
            self.compute_kl_divergence(text), self.reference_kl, self._kl_std
        )
        if self.method == "ppl":
            return ppl_score
        if self.method == "kl":
            return kl_score
        return 0.5 * (ppl_score + kl_score)

    def _deviation_score(
        self, value: float, reference: float | None, std: float | None
    ) -> float:
        """Map |value - reference| / scale through a sigmoid.

        The absolute value is intentional: both unusually high and unusually
        low statistics relative to the clean calibration set are anomalous.
        """
        if not math.isfinite(value):
            return 1.0
        if reference is None or reference == 0:
            return _sigmoid(value)
        scale = std if std not in (None, 0.0) else abs(reference)
        if scale == 0:
            scale = 1.0
        return _sigmoid(abs(value - reference) / scale)
