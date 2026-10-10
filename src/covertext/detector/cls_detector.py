"""LSGC classification-mode detector: frozen causal LM, trained linear head.

Wang et al., "Linguistic Steganalysis via LLMs", arXiv:2406.04218, convert a
causal LM into a sequence classifier. One forward pass produces the last-layer
hidden states (their E^L). A randomly initialized linear map sends the pooled
state to a two-class distribution. The backbone stays frozen: the encoders
share this model, and the paper's own classification mode adds the linear
layer on top of features from a single pass.

For a causal transformer the hidden state at position t depends only on tokens
0..t. The last non-padding position is therefore the only vector that has
seen the whole text. That is the pool used by causal sequence classification.

The head is trained by minimizing the mean cross-entropy

    L(W, b) = -(1/N) sum_i log softmax(W h_i + b)_{y_i}

with AdamW. Features are z-scored with training-set mean and standard
deviation before the linear map. An affine map on z-scored features is still
an affine classifier of the raw hidden state; the rescaling only makes the
step size meaningful. Those train statistics are stored and reapplied at
score time, so evaluation text never enters the normalizer.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from covertext.common.interfaces import Detector


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return hidden[:, last_real, :] with shape (batch, hidden).

    ``attention_mask`` is 1 on real tokens and 0 on padding. An all-zero row
    is an empty sequence and is rejected.
    """
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape (batch, time, hidden)")
    if attention_mask.shape[:2] != hidden.shape[:2]:
        raise ValueError("attention_mask batch and time must match hidden")
    lengths = attention_mask.to(dtype=torch.long).sum(dim=1)
    if torch.any(lengths < 1):
        raise ValueError("attention_mask marks an empty sequence")
    index = lengths - 1
    batch = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch, index]


def stego_probability(logits: torch.Tensor) -> torch.Tensor:
    """P(y = stego) from two-class logits. Column 1 is the stego class.

    Equals sigmoid(logit_stego - logit_cover), which is the two-class softmax.
    """
    if logits.shape[-1] != 2:
        raise ValueError("logits must have two classes, cover then stego")
    return torch.softmax(logits.float(), dim=-1)[..., 1]


class CLSDetector(Detector):
    """Linear stego classifier on the frozen LM's last-token hidden state."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        seed: int = 0,
        max_length: int = 256,
        lr: float = 1e-2,
        epochs: int = 30,
        batch_size: int = 16,
        weight_decay: float = 0.01,
    ):
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.seed = seed
        self.max_length = max_length
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.hidden_size = _hidden_size(model)
        self.head = _new_head(self.hidden_size, seed)
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        self._fitted = False

    @property
    def name(self) -> str:
        return "CLS"

    def sequence_features(self, texts: list[str]) -> torch.Tensor:
        """Last-layer, last-token hidden states, shape (len(texts), hidden).

        The transformer runs in chunks of ``batch_size``. Pooling happens
        inside each chunk, so padding never crosses a chunk boundary and the
        activation memory stays proportional to one chunk, not to the corpus.
        """
        if not texts:
            raise ValueError("texts must be non-empty")
        if any(not text or not text.strip() for text in texts):
            raise ValueError("texts must be non-empty strings")
        device = _device(self.model)
        body = getattr(self.model, "model", self.model)
        pooled_rows: list[torch.Tensor] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            input_ids, attention_mask = self._tokenize(chunk)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            with torch.no_grad():
                output = body(input_ids=input_ids, attention_mask=attention_mask)
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                hidden = output[0]
            pooled_rows.append(last_token_pool(hidden, attention_mask).float().cpu())
            del output, hidden, input_ids, attention_mask
        return torch.cat(pooled_rows, dim=0)

    def _tokenize(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad, and if needed keep the suffix so the stego tail is not dropped.

        Generative steganography writes the payload into the continuation, which
        sits at the end of the string. Tokenizer truncation defaults to dropping
        that suffix. Texts at or under ``max_length`` are unchanged.
        """
        encoded = self.tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)
        rows = [_as_token_row(row) for row in _as_rows(encoded["input_ids"])]
        if len(rows) != len(texts):
            raise ValueError("tokenizer returned a different number of rows than texts")
        trimmed: list[list[int]] = []
        for row in rows:
            if len(row) > self.max_length:
                row = row[-self.max_length :]
            if not row:
                raise ValueError("text tokenized to zero tokens")
            trimmed.append(row)
        width = max(len(row) for row in trimmed)
        input_ids = torch.zeros(len(trimmed), width, dtype=torch.long)
        attention_mask = torch.zeros(len(trimmed), width, dtype=torch.long)
        for index, row in enumerate(trimmed):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[index, : len(row)] = 1
        return input_ids, attention_mask

    def fit(self, texts: list[str], labels: list[int]) -> dict[str, float]:
        """Train the linear head. Reinitializes weights so each call is independent.

        ``labels`` are 0 for cover and 1 for stego. Both classes are required.
        Returns the cross-entropy on the training features before the first
        step and after the last step.
        """
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        if len(texts) < 2:
            raise ValueError("fit requires at least two labeled texts")
        if any(label not in (0, 1) for label in labels):
            raise ValueError("labels must be 0 (cover) or 1 (stego)")
        if len(set(labels)) < 2:
            raise ValueError("fit requires both cover and stego labels")

        features = self.sequence_features(texts)
        targets = torch.tensor(labels, dtype=torch.long)
        self._mean = features.mean(dim=0)
        self._std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
        standardized = self._standardize(features)

        self.head = _new_head(self.hidden_size, self.seed)
        loss_before = _cross_entropy(self.head, standardized, targets)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        optimizer = torch.optim.AdamW(
            self.head.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        order = torch.arange(standardized.size(0))
        for _ in range(self.epochs):
            perm = order[torch.randperm(order.numel(), generator=generator)]
            for start in range(0, perm.numel(), self.batch_size):
                batch = perm[start : start + self.batch_size]
                loss = F.cross_entropy(self.head(standardized[batch]), targets[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self._fitted = True
        loss_after = _cross_entropy(self.head, standardized, targets)
        return {"loss_before": loss_before, "loss_after": loss_after, "n": float(len(texts))}

    def score(self, text: str, **kwargs) -> float:
        """Return P(stego | text) in (0, 1)."""
        if not text or not text.strip():
            raise ValueError("text must be non-empty")
        if not self._fitted or self._mean is None or self._std is None:
            raise ValueError("CLSDetector must be fit before scoring")
        features = self._standardize(self.sequence_features([text]))
        with torch.no_grad():
            probability = stego_probability(self.head(features))[0]
        value = float(probability)
        if not math.isfinite(value):
            raise ValueError("classifier produced a non-finite probability")
        return min(1.0, max(0.0, value))

    def save(self, path: str) -> None:
        if not self._fitted or self._mean is None or self._std is None:
            raise ValueError("CLSDetector must be fit before saving")
        torch.save(
            {
                "weight": self.head.weight.detach().cpu(),
                "bias": self.head.bias.detach().cpu(),
                "mean": self._mean.cpu(),
                "std": self._std.cpu(),
                "hidden_size": self.hidden_size,
                "seed": self.seed,
            },
            path,
        )

    def load(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if int(state["hidden_size"]) != self.hidden_size:
            raise ValueError("saved head hidden size does not match this model")
        self.head = nn.Linear(self.hidden_size, 2)
        with torch.no_grad():
            self.head.weight.copy_(state["weight"])
            self.head.bias.copy_(state["bias"])
        self._mean = state["mean"].float()
        self._std = state["std"].float()
        self._fitted = True

    def _standardize(self, features: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._std is None:
            raise ValueError("standardizer is not fit")
        return (features - self._mean) / self._std


def _as_rows(raw) -> list:
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if raw and isinstance(raw[0], int):
        return [raw]
    return list(raw)


def _as_token_row(row) -> list[int]:
    if hasattr(row, "tolist"):
        row = row.tolist()
    return [int(token) for token in row]


def _hidden_size(model: PreTrainedModel) -> int:
    config = getattr(model, "config", None)
    size = getattr(config, "hidden_size", None)
    if not isinstance(size, int) or size < 1:
        raise ValueError("model.config.hidden_size is required")
    return size


def _device(model: PreTrainedModel) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return device if isinstance(device, torch.device) else torch.device(device)
    body = getattr(model, "model", model)
    try:
        return next(body.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _new_head(hidden_size: int, seed: int) -> nn.Linear:
    """Kaiming-uniform linear map, seeded, bias zero.

    PyTorch's default Linear init is U(-1/sqrt(fan_in), 1/sqrt(fan_in)),
    which is the uniform Kaiming bound for a linear layer with leaky_relu
    gain. Replaying it from ``seed`` keeps a refit reproducible.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    bound = 1.0 / math.sqrt(hidden_size)
    weight = torch.empty(2, hidden_size).uniform_(-bound, bound, generator=generator)
    head = nn.Linear(hidden_size, 2)
    with torch.no_grad():
        head.weight.copy_(weight)
        head.bias.zero_()
    return head


def _cross_entropy(head: nn.Linear, features: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        return float(F.cross_entropy(head(features), targets))
