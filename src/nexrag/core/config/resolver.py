"""
Resolves dotted class path strings from .yaml into validated instances.

This is how YAML-driven extensibility works:
    "myproject.chunkers.MySemanticChunker"
        → importlib.import_module("myproject.chunkers")
        → getattr(module, "MySemanticChunker")
        → validate: issubclass(cls, BaseChunker)
        → cls(**params)
        → instance injected into pipeline stage

Every failure produces a ClassResolutionError with a clear, actionable message.
No silent failures, no generic ImportError bubbling up to users.
"""

from __future__ import annotations

import importlib
from typing import Any, TypeVar

from nexrag.exceptions import ClassResolutionError

T = TypeVar("T")


def resolve_class(
    class_path: str,
    expected_base: type[T],
    params: dict[str, Any] | None = None,
    *,
    stage: str = "config",
    component: str = "resolver",
) -> T:
    """
    Import and instantiate a class from a dotted path string.

    Args:
        class_path:    Dotted path to the class. e.g. "myproject.chunkers.MyChunker"
        expected_base: The abstract base class the resolved class must extend.
                       e.g. BaseChunker, BaseLLM, BaseEmbedder
        params:        Optional dict of kwargs to pass to the class __init__.
        stage:         Stage name for error context.
        component:     Component name for error context.

    Returns:
        An instantiated instance of the resolved class.

    Raises:
        ClassResolutionError: If the class cannot be imported, is not found,
                              or does not extend expected_base.
    """
    params = params or {}

    module_path, class_name = _split_class_path(class_path)
    module = _import_module(module_path, class_path, stage, component)
    cls = _get_class(module, class_name, class_path, stage, component)
    _validate_base(cls, expected_base, class_path, stage, component)
    return _instantiate(cls, params, class_path, stage, component)


def _split_class_path(
    class_path: str,
) -> tuple[str, str]:
    """Split "a.b.c.MyClass" into ("a.b.c", "MyClass")."""
    if "." not in class_path:
        raise ClassResolutionError(
            f"Invalid class path: '{class_path}'. "
            f"Must be a dotted path including the module: "
            f"e.g. 'myproject.chunkers.MyChunker'",
            stage="config",
            component="resolver",
        )
    module_path, class_name = class_path.rsplit(".", 1)
    return module_path, class_name


def _import_module(
    module_path: str,
    class_path: str,
    stage: str,
    component: str,
) -> Any:
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ClassResolutionError(
            f"Module '{module_path}' not found (from class path '{class_path}'). "
            f"Ensure the package is installed and the path is correct.",
            stage=stage,
            component=component,
            cause=e,
        ) from e
    except ImportError as e:
        raise ClassResolutionError(
            f"Failed to import module '{module_path}': {e}",
            stage=stage,
            component=component,
            cause=e,
        ) from e


def _get_class(
    module: Any,
    class_name: str,
    class_path: str,
    stage: str,
    component: str,
) -> type:
    cls = getattr(module, class_name, None)
    if cls is None:
        available = [name for name in dir(module) if not name.startswith("_")]
        raise ClassResolutionError(
            f"Class '{class_name}' not found in module '{module.__name__}'. "
            f"Available names: {', '.join(available[:10])}",
            stage=stage,
            component=component,
        )
    if not isinstance(cls, type):
        raise ClassResolutionError(
            f"'{class_path}' is not a class (got {type(cls).__name__}). "
            f"The class path must point to a class, not a function or variable.",
            stage=stage,
            component=component,
        )
    return cls


def _validate_base(
    cls: type,
    expected_base: type,
    class_path: str,
    stage: str,
    component: str,
) -> None:
    if not issubclass(cls, expected_base):
        raise ClassResolutionError(
            f"'{class_path}' does not extend {expected_base.__name__}. "
            f"Your class must inherit from nexrag.core.interfaces.{expected_base.__name__}. "
            f"Example: class {cls.__name__}({expected_base.__name__}): ...",
            stage=stage,
            component=component,
        )


def _instantiate(
    cls: type,
    params: dict[str, Any],
    class_path: str,
    stage: str,
    component: str,
) -> Any:
    try:
        return cls(**params)
    except TypeError as e:
        raise ClassResolutionError(
            f"Failed to instantiate '{class_path}' with params {params}: {e}. "
            f"Check that your __init__ accepts these keyword arguments.",
            stage=stage,
            component=component,
            cause=e,
        ) from e
    except Exception as e:
        raise ClassResolutionError(
            f"'{class_path}.__init__' raised an unexpected error: {e}",
            stage=stage,
            component=component,
            cause=e,
        ) from e
