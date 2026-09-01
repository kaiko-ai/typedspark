# mypy: warn_unused_ignores=True
# pyright: reportUnnecessaryTypeIgnoreComment=error
"""Type-checker assertions for ``DataSet`` and ``DataSetWith``.

This module is intentionally **not** named ``test_*``: pytest does not collect
it. It exists to be exercised by ``mypy`` and ``pyright`` (which both run on
the whole repo in CI):

* Positive cases use :func:`typing_extensions.assert_type` to lock in inferred
  types. If the inferred type ever changes, both checkers fail.
* Negative cases mark expected errors with ``# type: ignore[<code>]``. The
  file-level pragmas above turn unused ignores into errors, so if a line ever
  stops being a type error (e.g. because variance was relaxed), the ignore
  becomes "unused" and CI fails.

The module is import-safe — it never executes anything that touches Spark — so
mypy/pyright can analyze it without a runtime environment.
"""

from __future__ import annotations

from typing import Protocol

from pyspark.sql.types import LongType, StringType
from typing_extensions import assert_type

from typedspark import Column, DataSet, DataSetWith, Schema


class _Age(Schema, Protocol):
    age: Column[LongType]


class _Person(Schema):
    name: Column[StringType]
    age: Column[LongType]


def _person() -> DataSet[_Person]:  # pragma: no cover - type-only helper
    raise NotImplementedError


# ---------- positive cases ----------

# A ``DataSet[T]`` is a ``DataSetWith[T]``: this assignment must type-check.
_self: DataSetWith[_Person] = _person()

# Covariance: ``_Person`` structurally implements the ``_Age`` protocol, so
# ``DataSet[_Person]`` must be assignable to ``DataSetWith[_Age]``.
_widened: DataSetWith[_Age] = _person()


def needs_age(df: DataSetWith[_Age]) -> DataSet[_Age]:
    """A function written against the wider protocol can take any DataSet whose
    schema covers ``_Age``."""
    return DataSet[_Age](df.select(_Age.age))


# Calling the function with a more specific DataSet must type-check.
_projected = needs_age(_person())
assert_type(_projected, DataSet[_Age])


# ---------- negative cases ----------

# ``DataSet`` is invariant in its schema. Even though ``_Person`` extends
# ``_Age`` structurally, ``DataSet[_Person]`` is **not** a ``DataSet[_Age]``;
# users must opt into the wider protocol via ``DataSetWith``.
_strict: DataSet[_Age] = _person()  # type: ignore[assignment]


def needs_age_strict(df: DataSet[_Age]) -> None:
    """Strict variant — only accepts ``DataSet[_Age]`` exactly."""


# Passing a ``DataSet[_Person]`` to the strict signature must error.
needs_age_strict(_person())  # type: ignore[arg-type]
