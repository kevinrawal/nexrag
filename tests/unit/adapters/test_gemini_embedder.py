from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("google.genai")

from nexrag.exceptions import EmbedderError


class _FakeAPIError(Exception):
    def __init__(self, code: int, message: str = "error") -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class _FakeServerError(_FakeAPIError):
    pass


def _patch_errors():
    return patch.multiple(
        "google.genai.errors",
        APIError=_FakeAPIError,
        ServerError=_FakeServerError,
    )


def _make_embedder(vectors=None, **kwargs):
    from nexrag.adapters.embedders.gemini import GeminiEmbedder

    vectors = vectors if vectors is not None else [[0.1, 0.2, 0.3]]
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = SimpleNamespace(
        embeddings=[SimpleNamespace(values=v) for v in vectors]
    )
    with patch("google.genai.Client", return_value=mock_client):
        embedder = GeminiEmbedder(model="gemini-embedding-001", api_key="test-key", **kwargs)
    embedder._client = mock_client
    return embedder


class TestGeminiEmbedder:
    def test_embed_returns_vectors(self):
        embedder = _make_embedder([[1.0, 2.0], [3.0, 4.0]])
        out = embedder.embed(["a", "b"])
        assert out == [[1.0, 2.0], [3.0, 4.0]]

    def test_embed_empty_returns_empty(self):
        embedder = _make_embedder()
        assert embedder.embed([]) == []
        embedder._client.models.embed_content.assert_not_called()

    def test_embed_query_returns_single_vector(self):
        embedder = _make_embedder([[0.5, 0.6, 0.7]])
        assert embedder.embed_query("hi") == [0.5, 0.6, 0.7]

    def test_dimensions_detected_lazily(self):
        embedder = _make_embedder([[0.1, 0.2, 0.3, 0.4]])
        assert embedder.dimensions == 4

    def test_model_name(self):
        assert _make_embedder().model_name == "gemini-embedding-001"

    def test_batching_makes_multiple_calls(self):
        from nexrag.adapters.embedders.gemini import GeminiEmbedder

        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = lambda model, contents: SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1, 0.2]) for _ in contents]
        )
        with patch("google.genai.Client", return_value=mock_client):
            embedder = GeminiEmbedder(api_key="k", batch_size=2)
        embedder._client = mock_client

        out = embedder.embed(["a", "b", "c", "d", "e"])
        assert len(out) == 5
        assert mock_client.models.embed_content.call_count == 3  # 2 + 2 + 1

    def test_length_mismatch_raises(self):
        embedder = _make_embedder([[1.0, 2.0]])  # one vector returned...
        with pytest.raises(EmbedderError):
            embedder.embed(["a", "b"])  # ...for two inputs

    def test_rate_limit_raises_embedder_error(self):
        embedder = _make_embedder()
        embedder._client.models.embed_content.side_effect = _FakeAPIError(429, "rate")
        with _patch_errors(), pytest.raises(EmbedderError):
            embedder.embed_query("q")

    def test_retries_on_server_error_then_succeeds(self):
        embedder = _make_embedder([[1.0, 2.0]])
        good = embedder._client.models.embed_content.return_value
        embedder._client.models.embed_content.side_effect = [_FakeServerError(503), good]
        with _patch_errors(), patch("time.sleep"):
            out = embedder.embed_query("q")
        assert out == [1.0, 2.0]
