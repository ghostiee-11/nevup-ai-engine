from unittest.mock import AsyncMock, patch

from app.memory.embeddings import _fallback_embedding, embed, embed_batch


async def test_embed_returns_768d_vector():
    fake = {"embedding": [0.1] * 768}
    with patch("app.memory.embeddings._configure", return_value=True), \
         patch("app.memory.embeddings._embed_call", AsyncMock(return_value=fake)):
        v = await embed("hello world")
    assert isinstance(v, list)
    assert len(v) == 768


async def test_embed_batch_returns_one_vector_per_input():
    fake = {"embedding": [0.1] * 768}
    with patch("app.memory.embeddings._configure", return_value=True), \
         patch("app.memory.embeddings._embed_call", AsyncMock(return_value=fake)):
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

    with patch("app.memory.embeddings._configure", return_value=True), \
         patch("app.memory.embeddings._embed_call", side_effect=flaky):
        v = await embed("retry me")
    assert calls["n"] == 2
    assert len(v) == 768


async def test_embed_falls_back_when_no_key():
    """When Gemini key is missing, embed must return a deterministic fallback vector
    rather than raising. The brief mandates fallbacks for every external call.
    """
    with patch("app.memory.embeddings._configure", return_value=False):
        v = await embed("anything")
    assert len(v) == 768
    assert all(-1.0 <= x <= 1.0 for x in v)


async def test_embed_falls_back_when_all_retries_fail():
    async def always_fail(_):
        raise RuntimeError("gemini down")

    with patch("app.memory.embeddings._configure", return_value=True), \
         patch("app.memory.embeddings._embed_call", side_effect=always_fail):
        v = await embed("anything")
    assert len(v) == 768  # fallback kicks in after retries exhausted


def test_fallback_embedding_is_deterministic():
    a = _fallback_embedding("hello world")
    b = _fallback_embedding("hello world")
    c = _fallback_embedding("different text")
    assert a == b
    assert a != c
    assert len(a) == 768
