"""Shared pytest fixtures. Model/dataset fixtures land in later issues."""

import pytest

from covertext.common.model import load_model


@pytest.fixture(scope="session")
def model_fixture():
    model, tokenizer = load_model()
    return model, tokenizer
