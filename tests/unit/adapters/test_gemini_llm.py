import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("google.genai")

from nexrag.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError

# --- Fake error classes patched onto the real google.genai.errors module so the
# exception-mapping tests don't depend on the SDK's error constructor signatures
# (which vary across versions). isinstance checks in the adapter resolve to these. ---


class _FakeAPIError(Exception):
    def __init__(self, code: int, message: str = "error") -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class _FakeClientError(_FakeAPIError):
    pass


class _FakeServerError(_FakeAPIError):
    pass


def _patch_errors():
    return patch.multiple(
        "google.genai.errors",
        APIError=_FakeAPIError,
        ClientError=_FakeClientError,
        ServerError=_FakeServerError,
    )


def _make_llm(response_text: str | None = "Answer.", with_usage: bool = True, **kwargs):
    from nexrag.adapters.llms.gemini import GeminiLLM

    mock_client = MagicMock()
    resp = MagicMock()
    resp.text = response_text
    if with_usage:
        resp.usage_metadata.prompt_token_count = 10
        resp.usage_metadata.candidates_token_count = 5
        resp.usage_metadata.total_token_count = 15
    else:
        resp.usage_metadata = None
    mock_client.models.generate_content.return_value = resp

    with patch("google.genai.Client", return_value=mock_client):
        llm = GeminiLLM(model="gemini-2.5-flash", api_key="test-key", **kwargs)
    llm._client = mock_client
    return llm


class TestGeminiLLM:
    def test_generate_returns_text_and_usage(self):
        llm = _make_llm("This is the answer.")
        text, usage = llm.generate("What is NexRAG?")
        assert text == "This is the answer."
        assert usage is not None
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15

    def test_generate_splits_system_and_user(self):
        llm = _make_llm("ok")
        llm.generate("System message\n\n---\n\nUser message")
        call = llm._client.models.generate_content.call_args
        assert call.kwargs["contents"] == "User message"
        assert call.kwargs["config"].system_instruction == "System message"

    def test_generate_no_separator_has_no_system_instruction(self):
        llm = _make_llm("ok")
        llm.generate("Just a user message")
        call = llm._client.models.generate_content.call_args
        assert call.kwargs["contents"] == "Just a user message"
        assert call.kwargs["config"].system_instruction is None

    def test_empty_response_returns_empty_string(self):
        llm = _make_llm(None)
        text, _ = llm.generate("q")
        assert text == ""

    def test_missing_usage_metadata_returns_none(self):
        llm = _make_llm("answer", with_usage=False)
        _, usage = llm.generate("q")
        assert usage is None

    def test_rate_limit_raises_llm_rate_limit_error(self):
        llm = _make_llm()
        llm._client.models.generate_content.side_effect = _FakeClientError(429, "rate limited")
        with _patch_errors(), pytest.raises(LLMRateLimitError):
            llm.generate("q")

    def test_auth_error_raises_llm_error(self):
        llm = _make_llm()
        llm._client.models.generate_content.side_effect = _FakeClientError(403, "forbidden")
        with _patch_errors(), pytest.raises(LLMError):
            llm.generate("q")

    def test_timeout_raises_llm_timeout_error(self):
        llm = _make_llm()
        llm._client.models.generate_content.side_effect = RuntimeError("request timed out")
        with _patch_errors(), pytest.raises(LLMTimeoutError):
            llm.generate("q")

    def test_generic_error_raises_llm_error(self):
        llm = _make_llm()
        llm._client.models.generate_content.side_effect = RuntimeError("network down")
        with _patch_errors(), pytest.raises(LLMError):
            llm.generate("q")

    def test_stream_yields_tokens_and_skips_empty(self):
        llm = _make_llm()

        def _chunk(text):
            c = MagicMock()
            c.text = text
            return c

        llm._client.models.generate_content_stream.return_value = iter(
            [_chunk("Hello"), _chunk(""), _chunk(" world")]
        )
        tokens = list(llm.stream("prompt"))
        assert "".join(tokens) == "Hello world"

    def test_async_generate_returns_text_and_usage(self):
        llm = _make_llm("Async answer.")
        resp = llm._client.models.generate_content.return_value
        llm._client.aio.models.generate_content = AsyncMock(return_value=resp)

        text, usage = asyncio.run(llm.async_generate("q"))
        assert text == "Async answer."
        assert usage is not None

    def test_async_stream_yields_tokens(self):
        llm = _make_llm()

        def _chunk(text):
            c = MagicMock()
            c.text = text
            return c

        async def _fake_stream():
            for token in ["Hello", " ", "world"]:
                yield _chunk(token)

        llm._client.aio.models.generate_content_stream = AsyncMock(return_value=_fake_stream())

        async def _collect():
            return [token async for token in llm.async_stream("prompt")]

        tokens = asyncio.run(_collect())
        assert "".join(tokens) == "Hello world"

    def test_split_prompt_with_separator(self):
        from nexrag.adapters.llms.gemini import GeminiLLM

        system, contents = GeminiLLM._split_prompt("sys\n\n---\n\nuser")
        assert system == "sys"
        assert contents == "user"

    def test_split_prompt_without_separator(self):
        from nexrag.adapters.llms.gemini import GeminiLLM

        system, contents = GeminiLLM._split_prompt("just user")
        assert system is None
        assert contents == "just user"

    def test_retries_on_server_error_then_succeeds(self):
        llm = _make_llm("Answer.")
        good = llm._client.models.generate_content.return_value
        llm._client.models.generate_content.side_effect = [
            _FakeServerError(503, "unavailable"),
            good,
        ]
        with _patch_errors(), patch("time.sleep"):
            text, _ = llm.generate("hello")
        assert text == "Answer."

    def test_retries_exhausted_raises(self):
        llm = _make_llm()
        llm._client.models.generate_content.side_effect = _FakeClientError(429, "rate")
        with _patch_errors(), patch("time.sleep"), pytest.raises(LLMRateLimitError):
            llm.generate("hello")

    def test_auth_error_not_retried(self):
        llm = _make_llm()
        llm._client.models.generate_content.side_effect = _FakeClientError(401, "bad key")
        with _patch_errors(), patch("time.sleep") as mock_sleep, pytest.raises(LLMError):
            llm.generate("hello")
        mock_sleep.assert_not_called()

    def test_max_retries_zero_never_retries(self):
        llm = _make_llm(max_retries=0)
        llm._client.models.generate_content.side_effect = _FakeServerError(503, "unavailable")
        with _patch_errors(), patch("time.sleep") as mock_sleep, pytest.raises(LLMError):
            llm.generate("hello")
        mock_sleep.assert_not_called()
        assert llm._client.models.generate_content.call_count == 1
