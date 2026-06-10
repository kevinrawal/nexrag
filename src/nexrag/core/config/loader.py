"""
Reads .yaml (user defined), resolves ${ENV_VAR} substitutions, validates against
the Pydantic schema, and returns a typed NexRAGConfig.

This is the only place in NexRAG that reads from disk for configuration.
Everything downstream receives a typed NexRAGConfig — never a raw dict.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from nexrag.core.config.schema import NexRAGConfig
from nexrag.exceptions import ConfigError

# Matches ${VAR_NAME} or ${VAR_NAME:-default_value}
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def load_config(path: str | Path) -> NexRAGConfig:
    """
    Load and validate a .yaml file.

    Steps:
        1. Read the YAML file from disk.
        2. Resolve all ${ENV_VAR} and ${ENV_VAR:-default} substitutions.
        3. Validate against the Pydantic schema.
        4. Return a fully typed NexRAGConfig.

    Args:
        path: Path to .yaml file (absolute or relative to cwd).

    Returns:
        Validated NexRAGConfig ready to pass to pipeline orchestrators.

    Raises:
        ConfigError: If the file is missing, unparseable, or fails validation.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise ConfigError(
            f"Configuration file not found: {config_path.resolve()}. "
            f"Create a nexrag.yaml or pass the correct path to NexRAG.from_config().",
            stage="config",
            component="loader",
        )

    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(
            f"Could not read configuration file: {config_path}",
            stage="config",
            component="loader",
            cause=e,
        ) from e

    # Resolve env vars before YAML parsing so substitutions work in any value.
    resolved_text = _resolve_env_vars(raw_text, config_path)

    try:
        raw_dict = yaml.safe_load(resolved_text)
    except yaml.YAMLError as e:
        raise ConfigError(
            f"Invalid YAML in {config_path}: {e}",
            stage="config",
            component="loader",
            cause=e,
        ) from e

    if not isinstance(raw_dict, dict):
        raise ConfigError(
            f"{config_path} must be a YAML mapping (dict) at the top level. "
            f"Got: {type(raw_dict).__name__}",
            stage="config",
            component="loader",
        )

    # Strip the top-level "nexrag:" wrapper if present.
    config_dict = raw_dict.get("nexrag", raw_dict)

    return _validate(config_dict, config_path)


def _resolve_env_vars(text: str, config_path: Path) -> str:
    """
    Replace ${VAR} and ${VAR:-default} patterns with environment variable values.

    ${VAR}           — raises ConfigError if VAR is not set.
    ${VAR:-default}  — uses "default" if VAR is not set.
    """

    def replacer(match: re.Match[str]) -> str:
        expression = match.group(1)

        if ":-" in expression:
            var_name, default_value = expression.split(":-", 1)
            value = os.environ.get(var_name.strip(), default_value)
        else:
            var_name = expression.strip()
            value = os.environ.get(var_name)
            if value is None:
                raise ConfigError(
                    f"Environment variable '{var_name}' is referenced in "
                    f"{config_path} but is not set. "
                    f"Set it in your environment or use ${{VAR:-default}} syntax.",
                    stage="config",
                    component="loader",
                )

        # Wrap in YAML single-quotes so special characters (:, #, [, {, |, >) don't
        # corrupt the YAML structure. Internal single quotes are escaped by doubling.
        return "'" + str(value).replace("'", "''") + "'"

    return _ENV_VAR_PATTERN.sub(replacer, text)


def _validate(config_dict: dict[str, Any], config_path: Path) -> NexRAGConfig:
    """Validate the config dict against the Pydantic schema."""
    try:
        return NexRAGConfig.model_validate(config_dict)
    except ValidationError as e:
        # Reformat Pydantic errors into human-readable ConfigErrors.
        errors = e.errors()
        messages = []
        for err in errors:
            field_path = " -> ".join(str(loc) for loc in err["loc"])
            messages.append(f"  {field_path}: {err['msg']}")
        formatted = "\n".join(messages)
        raise ConfigError(
            f"Invalid configuration in {config_path}:\n{formatted}",
            stage="config",
            component="loader",
            cause=e,
        ) from e
