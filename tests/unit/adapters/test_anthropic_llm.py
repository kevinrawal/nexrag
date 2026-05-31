from unittest.mock import MagicMock, patch

import pytest

from nexrag.exceptions import LLMError


def _make_mock_client(response_text: str = "Answer.") -> MagicMock:
    client = MagicMock()
    content_block = MagicMock()
    content_block.text = response_text
    client.messages.create.return_value = MagicMock(content=[content_block])
    return client


def _make_llm(response_text: str = "Answer.", **kwargs):
    from nexrag.adapters.llms.anthropic import AnthropicLLM

    mock_client = _make_mock_client(response_text)
    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = mock_client
    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        llm = AnthropicLLM(api_key="sk-ant-test", **kwargs)
    llm._client = mock_client
    return llm


def _make_fake_anthropic():
    """Return a fake anthropic module with real exception classes for isinstance checks."""
    fake = MagicMock()
    fake.RateLimitError = type("RateLimitError", (Exception,), {})
    fake.InternalServerError = type("InternalServerError", (Exception,), {})
    fake.APIConnectionError = type("APIConnectionError", (Exception,), {})
    fake.APITimeoutError = type("APITimeoutError", (Exception,), {})
    fake.AuthenticationError = type("AuthenticationError", (Exception,), {})
    return fake


class TestAnthropicLLM:
    def test_generate_returns_tuple(self):
        llm = _make_llm("This is the answer.")
        text, usage = llm.generate("What is NexRAG?")
        assert text == "This is the answer."

    def test_generate_prompt_with_separator_passes_system(self):
        llm = _make_llm("ok")
        llm.generate("System message\n\n---\n\nUser message")
        call_kwargs = llm._client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "System message"
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "User message"

    def test_generate_no_separator_has_no_system_key(self):
        llm = _make_llm("ok")
        llm.generate("Just a user message")
        call_kwargs = llm._client.messages.create.call_args.kwargs
        assert "system" not in call_kwargs
        assert call_kwargs["messages"][0]["role"] == "user"

    def test_system_content_stripped(self):
        llm = _make_llm("ok")
        llm.generate("  System  \n\n---\n\n  User  ")
        call_kwargs = llm._client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "System"
        assert call_kwargs["messages"][0]["content"] == "User"

    def test_empty_system_after_strip_omits_system_key(self):
        llm = _make_llm("ok")
        llm.generate("   \n\n---\n\nUser message")
        call_kwargs = llm._client.messages.create.call_args.kwargs
        assert "system" not in call_kwargs

    def test_max_tokens_passed_to_api(self):
        llm = _make_llm(max_tokens=512)
        llm.generate("q")
        call_kwargs = llm._client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 512

    def test_model_name_passed_to_api(self):
        llm = _make_llm(model="claude-3-haiku-20240307")
        llm.generate("q")
        call_kwargs = llm._client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-3-haiku-20240307"

    def test_generic_error_raises_llm_error(self):
        llm = _make_llm()
        llm._client.messages.create.side_effect = RuntimeError("network down")
        with pytest.raises(LLMError):
            llm.generate("q")

    def test_retries_on_rate_limit_then_succeeds(self):
        fake_anthropic = _make_fake_anthropic()
        mock_client = MagicMock()
        good_content = MagicMock()
        good_content.text = "Answer."
        good_resp = MagicMock()
        good_resp.content = [good_content]
        rate_err = fake_anthropic.RateLimitError("rate limited")
        mock_client.messages.create.side_effect = [rate_err, rate_err, good_resp]
        fake_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            from nexrag.adapters.llms.anthropic import AnthropicLLM

            llm = AnthropicLLM(api_key="sk-ant-test")
            llm._client = mock_client
            with patch("time.sleep"):
                text, _ = llm.generate("hello")
        assert text == "Answer."

    def test_retries_exhausted_raises_llm_rate_limit_error(self):
        fake_anthropic = _make_fake_anthropic()
        mock_client = MagicMock()
        rate_err = fake_anthropic.RateLimitError("rate limited")
        mock_client.messages.create.side_effect = [rate_err, rate_err, rate_err]
        fake_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            from nexrag.adapters.llms.anthropic import AnthropicLLM

            llm = AnthropicLLM(api_key="sk-ant-test")
            llm._client = mock_client
            with patch("time.sleep"):
                with pytest.raises(LLMError):
                    llm.generate("hello")

    def test_max_retries_zero_never_retries(self):
        fake_anthropic = _make_fake_anthropic()
        mock_client = MagicMock()
        rate_err = fake_anthropic.RateLimitError("rate limited")
        mock_client.messages.create.side_effect = rate_err
        fake_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            from nexrag.adapters.llms.anthropic import AnthropicLLM

            llm = AnthropicLLM(api_key="sk-ant-test", max_retries=0)
            llm._client = mock_client
            with patch("time.sleep") as mock_sleep:
                with pytest.raises(LLMError):
                    llm.generate("hello")
        mock_sleep.assert_not_called()
        assert mock_client.messages.create.call_count == 1

    def test_stream_yields_tokens(self):
        from nexrag.adapters.llms.anthropic import AnthropicLLM

        mock_client = MagicMock()
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Hello", " ", "world"])
        mock_client.messages.stream.return_value = mock_stream

        fake_anthropic = MagicMock()
        fake_anthropic.Anthropic.return_value = mock_client
        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            llm = AnthropicLLM(api_key="sk-ant-test")
        llm._client = mock_client

        tokens = list(llm.stream("prompt"))
        assert "".join(tokens) == "Hello world"

    def test_missing_anthropic_raises_llm_error(self):
        llm = _make_llm()
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises((LLMError, ImportError, AttributeError)):
                llm._build_client(None, None)

    def test_build_messages_with_separator(self):
        from nexrag.adapters.llms.anthropic import AnthropicLLM

        system, messages = AnthropicLLM._build_messages("sys\n\n---\n\nuser")
        assert system == "sys"
        assert messages == [{"role": "user", "content": "user"}]

    def test_build_messages_without_separator(self):
        from nexrag.adapters.llms.anthropic import AnthropicLLM

        system, messages = AnthropicLLM._build_messages("just user")
        assert system is None
        assert messages == [{"role": "user", "content": "just user"}]
