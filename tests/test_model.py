import pytest
import torch

from covertext.common.model import generate_completion, get_next_token_probs


@pytest.mark.slow
def test_load_model(model_fixture):
    model, tokenizer = model_fixture
    assert model is not None
    assert tokenizer is not None
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert str(model.device).startswith(expected)


@pytest.mark.slow
def test_get_next_token_probs(model_fixture):
    model, tokenizer = model_fixture
    probs, token_ids = get_next_token_probs(
        model, tokenizer, "The capital of France is"
    )
    assert probs.shape == token_ids.shape
    assert probs.dim() == 1
    assert torch.isclose(probs.sum().cpu(), torch.tensor(1.0), atol=1e-3)


@pytest.mark.slow
def test_generate_completion(model_fixture):
    model, tokenizer = model_fixture
    result = generate_completion(model, tokenizer, "Hello, my name is")
    assert isinstance(result, str)
    assert len(result) > 0
