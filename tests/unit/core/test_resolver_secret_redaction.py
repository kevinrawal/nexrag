import pytest

from nexrag.core.config.resolver import _redact, resolve_class
from nexrag.exceptions import ClassResolutionError


class TestRedact:
    def test_api_key_redacted(self):
        result = _redact({"api_key": "sk-secret123", "model": "gpt-4o"})
        assert result["api_key"] == "***"
        assert result["model"] == "gpt-4o"

    def test_token_redacted(self):
        assert _redact({"token": "tok-abc"})["token"] == "***"

    def test_secret_redacted(self):
        assert _redact({"secret": "mysecret"})["secret"] == "***"

    def test_password_redacted(self):
        assert _redact({"password": "hunter2"})["password"] == "***"

    def test_credential_redacted(self):
        assert _redact({"credential": "cred"})["credential"] == "***"

    def test_key_redacted(self):
        assert _redact({"key": "somekey"})["key"] == "***"

    def test_auth_redacted(self):
        assert _redact({"auth": "bearer xyz"})["auth"] == "***"

    def test_case_insensitive(self):
        result = _redact({"API_KEY": "sk-x", "Token": "abc"})
        assert result["API_KEY"] == "***"
        assert result["Token"] == "***"

    def test_non_sensitive_not_redacted(self):
        result = _redact({"model": "gpt-4o", "batch_size": 100})
        assert result["model"] == "gpt-4o"
        assert result["batch_size"] == 100

    def test_empty_dict(self):
        assert _redact({}) == {}


class TestResolveClassSecretLeakage:
    def test_bad_init_does_not_leak_api_key_in_error(self):
        """ClassResolutionError message must not contain the api_key value."""
        from nexrag.core.interfaces.chunker import BaseChunker

        # A class that exists but will fail to instantiate with these params
        class _BadChunker(BaseChunker):
            def chunk(self, document):  # type: ignore[override]
                return []

        import sys
        import types

        mod = types.ModuleType("_test_bad_chunker")
        mod._BadChunker = _BadChunker  # type: ignore[attr-defined]
        sys.modules["_test_bad_chunker"] = mod

        try:
            with pytest.raises(ClassResolutionError) as exc_info:
                resolve_class(
                    "_test_bad_chunker._BadChunker",
                    BaseChunker,
                    params={"api_key": "sk-supersecret", "unknown_param": 1},
                    stage="test",
                )
            assert "sk-supersecret" not in str(exc_info.value)
        finally:
            del sys.modules["_test_bad_chunker"]

    def test_invalid_class_path_raises_class_resolution_error(self):
        from nexrag.core.interfaces.chunker import BaseChunker

        with pytest.raises(ClassResolutionError, match="dotted"):
            resolve_class("NoDotsHere", BaseChunker)

    def test_missing_module_raises_class_resolution_error(self):
        from nexrag.core.interfaces.chunker import BaseChunker

        with pytest.raises(ClassResolutionError, match="not found"):
            resolve_class("nonexistent.module.MyClass", BaseChunker)

    def test_wrong_base_class_raises_class_resolution_error(self):
        import sys
        import types

        from nexrag.core.interfaces.chunker import BaseChunker
        from nexrag.core.interfaces.embedder import BaseEmbedder

        class _FakeChunker(BaseChunker):
            def chunk(self, document):  # type: ignore[override]
                return []

        mod = types.ModuleType("_test_wrong_base")
        mod._FakeChunker = _FakeChunker  # type: ignore[attr-defined]
        sys.modules["_test_wrong_base"] = mod

        try:
            with pytest.raises(ClassResolutionError, match="does not extend"):
                resolve_class("_test_wrong_base._FakeChunker", BaseEmbedder)
        finally:
            del sys.modules["_test_wrong_base"]
