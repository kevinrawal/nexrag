"""Tests for BaseLLM.generate() returning (text, TokenUsage | None) — issue #4."""

from unittest.mock import MagicMock

from nexrag.core.interfaces.llm import BaseLLM
from nexrag.core.models.metrics import TokenUsage


class ConcreteTestLLM(BaseLLM):
    """Minimal concrete implementation for testing BaseLLM."""

    def generate(self, prompt: str) -> tuple[str, TokenUsage | None]:
        return "test answer", None

    def stream(self, prompt):
        yield "test"


class TestBaseLLMGenerate:
    def test_returns_tuple(self):
        llm = ConcreteTestLLM()
        result = llm.generate("prompt")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_text_is_string(self):
        llm = ConcreteTestLLM()
        text, _ = llm.generate("prompt")
        assert isinstance(text, str)

    def test_usage_none_for_adapters_without_support(self):
        llm = ConcreteTestLLM()
        _, usage = llm.generate("prompt")
        assert usage is None


class TestOpenAILLMGenerate:
    def _make_response(self, content: str, prompt_tokens: int, completion_tokens: int):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = content
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = prompt_tokens
        resp.usage.completion_tokens = completion_tokens
        resp.usage.total_tokens = prompt_tokens + completion_tokens
        return resp

    def test_returns_text_and_token_usage(self):
        from nexrag.adapters.llms.openai import OpenAILLM

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_response(
            "GPT answer", 100, 50
        )

        llm = OpenAILLM.__new__(OpenAILLM)
        llm._model = "gpt-4o"
        llm._temperature = 0.2
        llm._max_tokens = 1024
        llm._timeout = 30
        llm._max_retries = 0
        llm._client = mock_client
        llm._async_client = MagicMock()

        text, usage = llm.generate("test prompt")

        assert text == "GPT answer"
        assert isinstance(usage, TokenUsage)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_usage_none_when_response_has_no_usage(self):
        from nexrag.adapters.llms.openai import OpenAILLM

        mock_client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "answer"
        resp.usage = None
        mock_client.chat.completions.create.return_value = resp

        llm = OpenAILLM.__new__(OpenAILLM)
        llm._model = "gpt-4o"
        llm._temperature = 0.2
        llm._max_tokens = 1024
        llm._timeout = 30
        llm._max_retries = 0
        llm._client = mock_client
        llm._async_client = MagicMock()

        text, usage = llm.generate("test prompt")
        assert text == "answer"
        assert usage is None


class TestAnthropicLLMGenerate:
    def _make_response(self, content: str, input_tokens: int, output_tokens: int):
        resp = MagicMock()
        resp.content = [MagicMock()]
        resp.content[0].text = content
        resp.usage = MagicMock()
        resp.usage.input_tokens = input_tokens
        resp.usage.output_tokens = output_tokens
        return resp

    def test_returns_text_and_token_usage(self):
        from nexrag.adapters.llms.anthropic import AnthropicLLM

        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_response("Claude answer", 200, 80)

        llm = AnthropicLLM.__new__(AnthropicLLM)
        llm._model = "claude-3-5-sonnet-20241022"
        llm._temperature = 0.2
        llm._max_tokens = 1024
        llm._timeout = 30
        llm._max_retries = 0
        llm._client = mock_client
        llm._async_client = MagicMock()

        text, usage = llm.generate("test prompt")

        assert text == "Claude answer"
        assert isinstance(usage, TokenUsage)
        assert usage.prompt_tokens == 200
        assert usage.completion_tokens == 80
        assert usage.total_tokens == 280


class TestOllamaLLMGenerate:
    def test_returns_text_and_none_usage(self):
        from nexrag.adapters.llms.ollama import OllamaLLM

        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "Ollama answer"}}

        llm = OllamaLLM.__new__(OllamaLLM)
        llm._model = "llama3.2"
        llm._base_url = "http://localhost:11434"
        llm._temperature = 0.2
        llm._max_tokens = 1024
        llm._timeout = 60
        llm._client = mock_client

        # Patch _build_client to return mock
        import unittest.mock as mock

        with mock.patch.object(llm, "_build_client", return_value=mock_client):
            text, usage = llm.generate("test prompt")

        assert text == "Ollama answer"
        assert usage is None
