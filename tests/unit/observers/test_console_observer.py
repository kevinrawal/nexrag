import json

from nexrag.core.models.event import PipelineEvent
from nexrag.observers.console import ConsoleObserver


def _event(status: str = "completed", stage: str = "embedder", duration_ms: float = 12.5):
    return PipelineEvent(
        pipeline_id="abc12345",
        stage=stage,
        status=status,
        duration_ms=duration_ms,
        metadata={"chunks": 5},
    )


class TestConsoleObserver:
    def test_info_level_emits_completed(self, capsys):
        obs = ConsoleObserver(log_level="INFO", format="json")
        obs.emit(_event("completed"))
        out = capsys.readouterr().out
        assert out.strip()

    def test_info_level_suppresses_started(self, capsys):
        obs = ConsoleObserver(log_level="INFO", format="json")
        obs.emit(_event("started"))
        out = capsys.readouterr().out
        assert out.strip() == ""

    def test_debug_level_emits_started(self, capsys):
        obs = ConsoleObserver(log_level="DEBUG", format="json")
        obs.emit(_event("started"))
        out = capsys.readouterr().out
        assert out.strip()

    def test_warning_level_suppresses_completed(self, capsys):
        obs = ConsoleObserver(log_level="WARNING", format="json")
        obs.emit(_event("completed"))
        out = capsys.readouterr().out
        assert out.strip() == ""

    def test_warning_level_emits_failed(self, capsys):
        obs = ConsoleObserver(log_level="WARNING", format="json")
        obs.emit(_event("failed"))
        out = capsys.readouterr().out
        assert out.strip()

    def test_json_format_is_valid_json(self, capsys):
        obs = ConsoleObserver(log_level="DEBUG", format="json")
        obs.emit(_event("completed"))
        out = capsys.readouterr().out.strip()
        data = json.loads(out)
        assert data["stage"] == "embedder"
        assert data["status"] == "completed"

    def test_json_format_includes_meta(self, capsys):
        obs = ConsoleObserver(log_level="DEBUG", format="json")
        obs.emit(_event("completed"))
        out = capsys.readouterr().out.strip()
        data = json.loads(out)
        assert "meta" in data
        assert data["meta"]["chunks"] == 5

    def test_json_format_includes_duration_ms(self, capsys):
        obs = ConsoleObserver(log_level="DEBUG", format="json")
        obs.emit(_event("completed", duration_ms=99.999))
        out = capsys.readouterr().out.strip()
        data = json.loads(out)
        assert data["duration_ms"] == 100.0

    def test_text_format_contains_stage(self, capsys):
        obs = ConsoleObserver(log_level="DEBUG", format="text")
        obs.emit(_event("completed"))
        out = capsys.readouterr().out
        assert "embedder" in out

    def test_text_format_contains_status(self, capsys):
        obs = ConsoleObserver(log_level="DEBUG", format="text")
        obs.emit(_event("failed"))
        out = capsys.readouterr().out
        assert "FAILED" in out

    def test_emit_never_raises(self):
        obs = ConsoleObserver(log_level="DEBUG", format="json")
        broken_event = object()  # not a PipelineEvent
        obs.emit(broken_event)  # type: ignore[arg-type]  — must not raise

    def test_no_config_values_in_output(self, capsys):
        obs = ConsoleObserver(log_level="DEBUG", format="json")
        obs.emit(_event("completed"))
        out = capsys.readouterr().out
        assert "api_key" not in out
        assert "sk-" not in out
