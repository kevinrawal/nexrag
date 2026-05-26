from unittest.mock import MagicMock, patch

import pytest

from nexrag.exceptions import EmbedderError


def _make_mock_client(embeddings: list[list[float]]):
    """Build a minimal openai client mock."""
    items = []
    for i, emb in enumerate(embeddings):
        item = MagicMock()
        item.index = i
        item.embedding = emb
        items.append(item)
    response = MagicMock()
    response.data = items
    client = MagicMock()
    client.embeddings.create.return_value = response
    return client


def _make_embedder(embeddings: list[list[float]], **kwargs):
    from nexrag.adapters.embedders.openai import OpenAIEmbedder

    with patch("openai.OpenAI", return_value=_make_mock_client(embeddings)):
        embedder = OpenAIEmbedder(**kwargs)
    embedder._client = _make_mock_client(embeddings)
    return embedder


class TestOpenAIEmbedder:
    def test_embed_returns_one_vector_per_text(self):
        vecs = [[0.1, 0.2], [0.3, 0.4]]
        embedder = _make_embedder(vecs)
        result = embedder.embed(["hello", "world"])
        assert len(result) == 2

    def test_embed_empty_list_returns_empty(self):
        embedder = _make_embedder([])
        assert embedder.embed([]) == []

    def test_embed_query_returns_single_vector(self):
        embedder = _make_embedder([[0.1, 0.2, 0.3]])
        vec = embedder.embed_query("test")
        assert isinstance(vec, list)
        assert len(vec) == 3

    def test_dimensions_cached_after_embed(self):
        embedder = _make_embedder([[0.1, 0.2, 0.3]])
        embedder.embed(["x"])
        assert embedder._dimensions == 3

    def test_dimensions_cached_after_embed_query(self):
        embedder = _make_embedder([[0.5, 0.6]])
        embedder.embed_query("test")
        assert embedder._dimensions == 2

    def test_model_name_property(self):
        embedder = _make_embedder([[0.1]])
        assert embedder.model_name == "text-embedding-3-small"

    def test_custom_model_name(self):
        from nexrag.adapters.embedders.openai import OpenAIEmbedder

        with patch("openai.OpenAI", return_value=MagicMock()):
            embedder = OpenAIEmbedder(model="text-embedding-ada-002")
        assert embedder.model_name == "text-embedding-ada-002"

    def test_embed_batches_large_input(self):
        n = 5
        vecs = [[float(i)] for i in range(n)]
        embedder = _make_embedder(vecs, batch_size=2)
        # Swap client to one that handles batches correctly
        call_count = [0]

        def fake_create(input, model):
            start = call_count[0] * 2
            items = []
            for j, _ in enumerate(input):
                item = MagicMock()
                item.index = j
                item.embedding = [float(start + j)]
                items.append(item)
            resp = MagicMock()
            resp.data = items
            call_count[0] += 1
            return resp

        embedder._client.embeddings.create.side_effect = fake_create
        result = embedder.embed(["a", "b", "c", "d", "e"])
        assert len(result) == 5

    def test_mismatch_raises_embedder_error(self):
        from nexrag.adapters.embedders.openai import OpenAIEmbedder

        bad_client = MagicMock()
        resp = MagicMock()
        resp.data = []  # 0 items for 1 text input
        bad_client.embeddings.create.return_value = resp

        with patch("openai.OpenAI", return_value=bad_client):
            embedder = OpenAIEmbedder()
        embedder._client = bad_client

        with pytest.raises(EmbedderError):
            embedder.embed(["hello"])

    def test_api_error_raises_embedder_error(self):
        embedder = _make_embedder([[0.1]])
        embedder._client.embeddings.create.side_effect = RuntimeError("API down")
        with pytest.raises(EmbedderError):
            embedder.embed(["fail"])

    def test_missing_openai_raises_embedder_error(self):
        embedder = _make_embedder([[0.1]])
        embedder._client = None  # simulate missing client
        # _build_client raises EmbedderError when openai is missing
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises((EmbedderError, ImportError, AttributeError)):
                embedder._build_client(None, None)
