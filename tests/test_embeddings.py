from unittest.mock import AsyncMock, patch

import pytest

from app.memory.embeddings import (
    EMBED_PATH_COUNTER,
    _fallback_embedding,
    embed,
    embed_batch,
)


@pytest.fixture(autouse=True)
def _reset_counters():
    for k in EMBED_PATH_COUNTER:
        EMBED_PATH_COUNTER[k] = 0
    yield


async def test_embed_returns_768d_vector_when_gemini_works():
    with patch("app.memory.embeddings._configure_gemini", return_value=True), \
         patch("app.memory.embeddings._embed_gemini", AsyncMock(return_value=[0.1] * 768)):
        v = await embed("hello world")
    assert isinstance(v, list)
    assert len(v) == 768
    assert EMBED_PATH_COUNTER["gemini"] == 1


async def test_embed_batch_returns_one_vector_per_input():
    with patch("app.memory.embeddings._configure_gemini", return_value=True), \
         patch("app.memory.embeddings._embed_gemini", AsyncMock(return_value=[0.1] * 768)):
        out = await embed_batch(["a", "b", "c"])
    assert len(out) == 3
    assert all(len(v) == 768 for v in out)
    assert EMBED_PATH_COUNTER["gemini"] == 3


async def test_embed_retries_gemini_on_transient_failure():
    calls = {"n": 0}

    async def flaky(_):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return [0.0] * 768

    with patch("app.memory.embeddings._configure_gemini", return_value=True), \
         patch("app.memory.embeddings._embed_gemini", side_effect=flaky):
        v = await embed("retry me")
    assert calls["n"] == 2
    assert len(v) == 768
    assert EMBED_PATH_COUNTER["gemini"] == 1


async def test_embed_falls_back_to_local_when_no_gemini_key():
    """Tier 2: Gemini disabled → fastembed local model is used."""
    fake_local = AsyncMock(return_value=[0.42] * 768)
    with patch("app.memory.embeddings._configure_gemini", return_value=False), \
         patch("app.memory.embeddings._embed_local", fake_local):
        v = await embed("anything")
    assert len(v) == 768
    assert EMBED_PATH_COUNTER["local"] == 1
    assert EMBED_PATH_COUNTER["gemini"] == 0


async def test_embed_falls_back_to_sha_when_local_fails():
    """Tier 3: both Gemini and fastembed fail → SHA pseudo-embedding (never raises)."""
    async def boom(_):
        raise RuntimeError("local broke")

    with patch("app.memory.embeddings._configure_gemini", return_value=False), \
         patch("app.memory.embeddings._embed_local", side_effect=boom):
        v = await embed("anything")
    assert len(v) == 768
    assert EMBED_PATH_COUNTER["sha"] == 1


async def test_embed_falls_back_through_chain_when_gemini_dies():
    async def gemini_dies(_):
        raise RuntimeError("gemini down")

    fake_local = AsyncMock(return_value=[0.5] * 768)
    with patch("app.memory.embeddings._configure_gemini", return_value=True), \
         patch("app.memory.embeddings._embed_gemini", side_effect=gemini_dies), \
         patch("app.memory.embeddings._embed_local", fake_local):
        v = await embed("anything")
    assert len(v) == 768
    assert EMBED_PATH_COUNTER["local"] == 1


def test_fallback_embedding_is_deterministic():
    a = _fallback_embedding("hello world")
    b = _fallback_embedding("hello world")
    c = _fallback_embedding("different text")
    assert a == b
    assert a != c
    assert len(a) == 768


@pytest.mark.integration
def test_local_embed_produces_semantic_similarity():
    """Real fastembed call: related texts must be closer than unrelated.
    This is what makes the local fallback actually useful (vs SHA which gives
    deterministic but meaningless vectors). Marked integration because it loads
    the ONNX model on first run (~80MB).
    """
    import math
    from app.memory.embeddings import _get_local_model

    model = _get_local_model()
    # 384d native; we don't pad here because cosine is invariant to padding.
    a = list(next(model.embed(["anxious revenge sequence after a losing close"])))
    b = list(next(model.embed(["trader chasing losses with anxious follow-on entries"])))
    c = list(next(model.embed(["calm planned exit per morning prep"])))

    def cos(x, y):
        dot = sum(xi * yi for xi, yi in zip(x, y))
        nx = math.sqrt(sum(xi * xi for xi in x))
        ny = math.sqrt(sum(yi * yi for yi in y))
        return dot / (nx * ny)

    sim_related = cos(a, b)
    sim_unrelated = cos(a, c)
    assert sim_related > sim_unrelated, (
        f"semantic broken: related={sim_related:.3f} not > unrelated={sim_unrelated:.3f}"
    )
