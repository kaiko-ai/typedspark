"""Helpers for keeping typedspark wrappers compatible with PySpark 4 dispatch."""

from __future__ import annotations

from typing import Any, Dict, Tuple, TypeVar, cast

_T = TypeVar("_T")

_CLASS_CACHE: Dict[Tuple[type, type], type] = {}


def _sanitize_name(name: str) -> str:
    """Return a safe ASCII identifier for dynamically generated class names."""
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def attach_mixin(instance: _T, mixin_cls: type) -> _T:
    """Attach a mixin class while keeping the original PySpark class in the MRO."""
    base_type = type(instance)
    if issubclass(base_type, mixin_cls):
        return instance

    key = (base_type, mixin_cls)
    new_cls = _CLASS_CACHE.get(key)
    if new_cls is None:
        mixin_name = _sanitize_name(getattr(mixin_cls, "__name__", "Mixin"))
        base_name = _sanitize_name(getattr(base_type, "__name__", "Base"))
        name = f"{mixin_name}With{base_name}"
        new_cls = type(name, (mixin_cls, base_type), {})
        _CLASS_CACHE[key] = new_cls

    # Pyright disallows __class__ assignment on typed objects; cast to Any to be explicit.
    cast(Any, instance).__class__ = new_cls
    return instance
