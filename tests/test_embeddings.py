from unittest.mock import AsyncMock, patch

import pytest

from app.memory.embeddings import embed, embed_batch


async def test_embed_returns_768d_vector():
    fake = {"embedding": [0.1] * 768}
    with patch("app.memory.embeddings._embed_call", AsyncMock(return_value=fake)):
        v = await embed("hello world")
    assert isinstance(v, list)
    assert len(v) == 768


async def test_embed_batch_returns_one_vector_per_input():
    fake = {"embedding": [0.1] * 768}
    with patch("app.memory.embeddings._embed_call", AsyncMock(return_value=fake)):
        out = await embed_batch(["a", "b", "c"])
    assert len(out) == 3
    assert all(len(v) == 768 for v in out)


async def test_embed_retries_on_transient_failure():
    calls = {"n": 0}

    async def flaky(_):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return {"embedding": [0.0] * 768}

    with patch("app.memory.embeddings._embed_call", side_effect=flaky):
        v = await embed("retry me")
    assert calls["n"] == 2
    assert len(v) == 768
