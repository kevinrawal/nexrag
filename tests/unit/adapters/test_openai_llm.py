from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openai")

from nexrag.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError


def _make_llm(response_text: str = "Answer."):
    from nexrag.adapters.llms.openai import OpenAILLM

    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = response_text
    mock_client.chat.completions.create.return_value = MagicMock(choices=[choice])

    with patch("openai.OpenAI", return_value=mock_client):
        llm = OpenAILLM(api_key="sk-test")
    llm._client = mock_client
    return llm


class TestOpenAILLM:
    def test_generate_returns_tuple(self):
        llm = _make_llm("This is the answer.")
        text, usage = llm.generate("What is NexRAG?")
        assert text == "This is the answer."

    def test_generate_prompt_split_into_system_user(self):
        llm = _make_llm("ok")
        llm.generate("System message\n\n---\n\nUser message")
        call_args = llm._client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_generate_no_separator_uses_user_role(self):
        llm = _make_llm("ok")
        llm.generate("Just a user message")
        messages = llm._client.chat.completions.create.call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_system_content_stripped(self):
        llm = _make_llm("ok")
        llm.generate("  System  \n\n---\n\n  User  ")
        messages = llm._client.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0]["content"] == "System"
        assert messages[1]["content"] == "User"

    def test_empty_response_returns_empty_string(self):
        llm = _make_llm(None)  # type: ignore[arg-type]
        llm._client.chat.completions.create.return_value.choices[0].message.content = None
        text, _ = llm.generate("q")
        assert text == ""

    def test_rate_limit_error_raises_llm_rate_limit_error(self):
        import openai

        llm = _make_llm()
        llm._client.chat.completions.create.side_effect = openai.RateLimitError(
            "rate limited", response=MagicMock(), body={}
        )
        with pytest.raises(LLMRateLimitError):
            llm.generate("q")

    def test_timeout_error_raises_llm_timeout_error(self):
        import openai

        llm = _make_llm()
        llm._client.chat.completions.create.side_effect = openai.APITimeoutError(
            request=MagicMock()
        )
        with pytest.raises(LLMTimeoutError):
            llm.generate("q")

    def test_auth_error_raises_llm_error(self):
        import openai

        llm = _make_llm()
        llm._client.chat.completions.create.side_effect = openai.AuthenticationError(
            "bad key", response=MagicMock(), body={}
        )
        with pytest.raises(LLMError):
            llm.generate("q")

    def test_generic_error_raises_llm_error(self):
        llm = _make_llm()
        llm._client.chat.completions.create.side_effect = RuntimeError("network down")
        with pytest.raises(LLMError):
            llm.generate("q")

    def test_stream_yields_tokens(self):
        from nexrag.adapters.llms.openai import OpenAILLM

        mock_client = MagicMock()
        chunks = []
        for token in ["Hello", " ", "world"]:
            c = MagicMock()
            c.choices[0].delta.content = token
            chunks.append(c)

        mock_client.chat.completions.create.return_value = iter(chunks)

        with patch("openai.OpenAI", return_value=mock_client):
            llm = OpenAILLM(api_key="sk-test")
        llm._client = mock_client

        tokens = list(llm.stream("prompt"))
        assert "".join(tokens) == "Hello world"

    def test_build_messages_with_separator(self):
        from nexrag.adapters.llms.openai import OpenAILLM

        msgs = OpenAILLM._build_messages("sys\n\n---\n\nuser")
        assert msgs[0] == {"role": "system", "content": "sys"}
        assert msgs[1] == {"role": "user", "content": "user"}

    def test_build_messages_without_separator(self):
        from nexrag.adapters.llms.openai import OpenAILLM

        msgs = OpenAILLM._build_messages("just user")
        assert msgs == [{"role": "user", "content": "just user"}]

    def test_retries_on_rate_limit_then_succeeds(self):
        import openai

        llm = _make_llm("Answer.")
        rate_err = openai.RateLimitError("rate limited", response=MagicMock(), body={})
        good_resp = llm._client.chat.completions.create.return_value
        llm._client.chat.completions.create.side_effect = [rate_err, rate_err, good_resp]

        with patch("time.sleep"):
            text, _ = llm.generate("hello")
        assert text == "Answer."

    def test_retries_exhausted_raises_llm_rate_limit_error(self):
        import openai

        llm = _make_llm()
        rate_err = openai.RateLimitError("rate limited", response=MagicMock(), body={})
        llm._client.chat.completions.create.side_effect = [rate_err, rate_err, rate_err]

        with patch("time.sleep"):
            with pytest.raises(LLMRateLimitError):
                llm.generate("hello")

    def test_auth_error_not_retried(self):
        import openai

        llm = _make_llm()
        auth_err = openai.AuthenticationError("bad key", response=MagicMock(), body={})
        llm._client.chat.completions.create.side_effect = auth_err

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(LLMError):
                llm.generate("hello")
        mock_sleep.assert_not_called()

    def test_max_retries_zero_never_retries(self):
        import openai

        from nexrag.adapters.llms.openai import OpenAILLM

        mock_client = MagicMock()
        rate_err = openai.RateLimitError("rate limited", response=MagicMock(), body={})
        mock_client.chat.completions.create.side_effect = rate_err

        with patch("openai.OpenAI", return_value=mock_client):
            llm = OpenAILLM(api_key="sk-test", max_retries=0)
        llm._client = mock_client

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(LLMRateLimitError):
                llm.generate("hello")
        mock_sleep.assert_not_called()
        assert llm._client.chat.completions.create.call_count == 1
