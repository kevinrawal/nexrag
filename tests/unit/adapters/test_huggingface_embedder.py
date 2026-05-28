from unittest.mock import MagicMock, patch

import pytest

from nexrag.exceptions import EmbedderError


def _make_response(vectors: list[list[float]]) -> MagicMock:
    """Wrap vectors in a mock that supports .tolist() like a numpy array."""
    resp = MagicMock()
    resp.tolist.return_value = vectors
    return resp


def _make_client(vectors: list[list[float]] | None = None) -> MagicMock:
    client = MagicMock()
    client.feature_extraction.return_value = _make_response(vectors or [[0.1, 0.2, 0.3]])
    return client


def _make_embedder(vectors: list[list[float]] | None = None, **kwargs):
    from nexrag.adapters.embedders.huggingface import HuggingFaceEmbedder

    mock_client = _make_client(vectors)
    fake_hf = MagicMock()
    fake_hf.InferenceClient.return_value = mock_client
    with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
        embedder = HuggingFaceEmbedder(**kwargs)
    embedder._client = mock_client
    return embedder


class TestHuggingFaceEmbedder:
    def test_embed_returns_one_vector_per_text(self):
        embedder = _make_embedder([[0.1, 0.2], [0.3, 0.4]])
        result = embedder.embed(["hello", "world"])
        assert len(result) == 2

    def test_embed_empty_list_returns_empty(self):
        embedder = _make_embedder()
        assert embedder.embed([]) == []

    def test_embed_query_returns_single_vector(self):
        embedder = _make_embedder([[0.1, 0.2, 0.3]])
        vec = embedder.embed_query("test")
        assert isinstance(vec, list)
        assert len(vec) == 3

    def test_embed_vectors_returned_as_lists(self):
        embedder = _make_embedder([[1.0, 2.0]])
        result = embedder.embed(["x"])
        assert isinstance(result[0], list)

    def test_dimensions_cached_after_embed(self):
        embedder = _make_embedder([[0.1, 0.2, 0.3]])
        embedder.embed(["x"])
        assert embedder._dimensions == 3

    def test_dimensions_cached_after_embed_query(self):
        embedder = _make_embedder([[0.5, 0.6]])
        embedder.embed_query("test")
        assert embedder._dimensions == 2

    def test_model_name_property(self):
        embedder = _make_embedder()
        assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_custom_model_name(self):
        embedder = _make_embedder(model="BAAI/bge-small-en-v1.5")
        assert embedder.model_name == "BAAI/bge-small-en-v1.5"

    def test_batch_size_respected(self):
        embedder = _make_embedder(batch_size=2)
        embedder._client.feature_extraction.side_effect = [
            _make_response([[0.1], [0.2]]),
            _make_response([[0.3]]),
        ]
        result = embedder.embed(["a", "b", "c"])
        assert embedder._client.feature_extraction.call_count == 2
        assert len(result) == 3

    def test_batch_preserves_order(self):
        embedder = _make_embedder(batch_size=2)
        embedder._client.feature_extraction.side_effect = [
            _make_response([[1.0], [2.0]]),
            _make_response([[3.0]]),
        ]
        result = embedder.embed(["a", "b", "c"])
        assert [v[0] for v in result] == [1.0, 2.0, 3.0]

    def test_auth_error_raises_embedder_error(self):
        embedder = _make_embedder()
        embedder._client.feature_extraction.side_effect = RuntimeError("401 Unauthorized")
        with pytest.raises(EmbedderError, match="authentication"):
            embedder.embed(["hello"])

    def test_model_not_found_raises_embedder_error(self):
        embedder = _make_embedder()
        embedder._client.feature_extraction.side_effect = RuntimeError(
            "model 'bad/model' not found"
        )
        with pytest.raises(EmbedderError, match="not found"):
            embedder.embed(["hello"])

    def test_generic_error_raises_embedder_error(self):
        embedder = _make_embedder()
        embedder._client.feature_extraction.side_effect = RuntimeError("connection error")
        with pytest.raises(EmbedderError):
            embedder.embed(["hello"])

    def test_missing_huggingface_hub_raises_embedder_error(self):
        embedder = _make_embedder()
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises((EmbedderError, ImportError, AttributeError)):
                embedder._build_client()

    def test_api_key_passed_as_token(self):
        from nexrag.adapters.embedders.huggingface import HuggingFaceEmbedder

        mock_client = _make_client()
        fake_hf = MagicMock()
        fake_hf.InferenceClient.return_value = mock_client
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            HuggingFaceEmbedder(api_key="hf-test-token")
        call_kwargs = fake_hf.InferenceClient.call_args.kwargs
        assert call_kwargs["token"] == "hf-test-token"

    def test_base_url_passed_through(self):
        from nexrag.adapters.embedders.huggingface import HuggingFaceEmbedder

        mock_client = _make_client()
        fake_hf = MagicMock()
        fake_hf.InferenceClient.return_value = mock_client
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            HuggingFaceEmbedder(base_url="https://my-endpoint.huggingface.cloud")
        call_kwargs = fake_hf.InferenceClient.call_args.kwargs
        assert call_kwargs["base_url"] == "https://my-endpoint.huggingface.cloud"

    def test_no_api_key_omits_token(self):
        from nexrag.adapters.embedders.huggingface import HuggingFaceEmbedder

        mock_client = _make_client()
        fake_hf = MagicMock()
        fake_hf.InferenceClient.return_value = mock_client
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            HuggingFaceEmbedder()
        call_kwargs = fake_hf.InferenceClient.call_args.kwargs
        assert "token" not in call_kwargs
