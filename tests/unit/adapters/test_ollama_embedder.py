from unittest.mock import MagicMock, patch

import pytest

from nexrag.exceptions import EmbedderError


def _make_client(embedding: list[float] | None = None) -> MagicMock:
    """Build a minimal ollama client mock."""
    client = MagicMock()
    client.embeddings.return_value = {"embedding": embedding or [0.1, 0.2, 0.3]}
    return client


def _make_embedder(embedding: list[float] | None = None, **kwargs):
    from nexrag.adapters.embedders.ollama import OllamaEmbedder

    mock_client = _make_client(embedding)
    # ollama may not be installed — inject a fake module so the lazy import succeeds
    fake_ollama = MagicMock()
    fake_ollama.Client.return_value = mock_client
    with patch.dict("sys.modules", {"ollama": fake_ollama}):
        embedder = OllamaEmbedder(**kwargs)
    embedder._client = mock_client
    return embedder


class TestOllamaEmbedder:
    def test_embed_returns_one_vector_per_text(self):
        embedder = _make_embedder([0.1, 0.2])
        embedder._client.embeddings.side_effect = [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
        ]
        result = embedder.embed(["hello", "world"])
        assert len(result) == 2

    def test_embed_empty_list_returns_empty(self):
        embedder = _make_embedder()
        assert embedder.embed([]) == []

    def test_embed_calls_api_once_per_text(self):
        embedder = _make_embedder([0.1, 0.2])
        embedder._client.embeddings.side_effect = [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
            {"embedding": [0.5, 0.6]},
        ]
        embedder.embed(["a", "b", "c"])
        assert embedder._client.embeddings.call_count == 3

    def test_embed_query_returns_single_vector(self):
        embedder = _make_embedder([0.1, 0.2, 0.3])
        vec = embedder.embed_query("test")
        assert isinstance(vec, list)
        assert len(vec) == 3

    def test_dimensions_cached_after_embed(self):
        embedder = _make_embedder([0.1, 0.2, 0.3])
        embedder.embed(["x"])
        assert embedder._dimensions == 3

    def test_dimensions_cached_after_embed_query(self):
        embedder = _make_embedder([0.5, 0.6])
        embedder.embed_query("test")
        assert embedder._dimensions == 2

    def test_model_name_property(self):
        embedder = _make_embedder()
        assert embedder.model_name == "nomic-embed-text"

    def test_custom_model_name(self):
        embedder = _make_embedder(model="mxbai-embed-large")
        assert embedder.model_name == "mxbai-embed-large"

    def test_base_url_trailing_slash_stripped(self):
        embedder = _make_embedder(base_url="http://localhost:11434/")
        assert embedder._base_url == "http://localhost:11434"

    def test_connection_error_raises_embedder_error(self):
        embedder = _make_embedder()
        embedder._client.embeddings.side_effect = ConnectionRefusedError("connection refused")
        with pytest.raises(EmbedderError, match="connect"):
            embedder.embed(["hello"])

    def test_model_not_found_raises_embedder_error(self):
        embedder = _make_embedder()
        embedder._client.embeddings.side_effect = RuntimeError(
            "model 'bad-model' not found, pull it first"
        )
        with pytest.raises(EmbedderError, match="not found"):
            embedder.embed(["hello"])

    def test_generic_error_raises_embedder_error(self):
        embedder = _make_embedder()
        embedder._client.embeddings.side_effect = RuntimeError("unexpected error")
        with pytest.raises(EmbedderError):
            embedder.embed(["hello"])

    def test_missing_ollama_raises_embedder_error(self):
        embedder = _make_embedder()
        with patch.dict("sys.modules", {"ollama": None}):
            with pytest.raises((EmbedderError, ImportError, AttributeError)):
                embedder._build_client()

    def test_vectors_returned_as_lists(self):
        embedder = _make_embedder([1.0, 2.0, 3.0])
        result = embedder.embed(["x"])
        assert isinstance(result[0], list)

    def test_embed_preserves_order(self):
        embedder = _make_embedder()
        embedder._client.embeddings.side_effect = [
            {"embedding": [1.0]},
            {"embedding": [2.0]},
            {"embedding": [3.0]},
        ]
        result = embedder.embed(["a", "b", "c"])
        assert [v[0] for v in result] == [1.0, 2.0, 3.0]
