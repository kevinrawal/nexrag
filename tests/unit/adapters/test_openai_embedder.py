from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openai")

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

    def test_retries_on_rate_limit_then_succeeds(self):
        import openai

        embedder = _make_embedder([[0.1, 0.2]])
        rate_err = openai.RateLimitError("rate limited", response=MagicMock(), body={})
        # Fail twice, succeed on third attempt
        good_resp = embedder._client.embeddings.create.return_value
        embedder._client.embeddings.create.side_effect = [rate_err, rate_err, good_resp]

        with patch("time.sleep"):  # don't actually wait
            result = embedder.embed(["hello"])
        assert len(result) == 1

    def test_retries_exhausted_raises_embedder_error(self):
        import openai

        embedder = _make_embedder([[0.1]])
        rate_err = openai.RateLimitError("rate limited", response=MagicMock(), body={})
        embedder._client.embeddings.create.side_effect = [rate_err, rate_err, rate_err]

        with patch("time.sleep"):
            with pytest.raises(EmbedderError):
                embedder.embed(["hello"])

    def test_auth_error_not_retried(self):
        import openai

        embedder = _make_embedder([[0.1]])
        auth_err = openai.AuthenticationError("bad key", response=MagicMock(), body={})
        embedder._client.embeddings.create.side_effect = auth_err

        # Should raise immediately — no sleep
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(EmbedderError):
                embedder.embed(["hello"])
        mock_sleep.assert_not_called()

    def test_max_retries_zero_never_retries(self):
        import openai

        embedder = _make_embedder([[0.1]], max_retries=0)
        rate_err = openai.RateLimitError("rate limited", response=MagicMock(), body={})
        embedder._client.embeddings.create.side_effect = rate_err

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(EmbedderError):
                embedder.embed(["hello"])
        mock_sleep.assert_not_called()
        assert embedder._client.embeddings.create.call_count == 1
