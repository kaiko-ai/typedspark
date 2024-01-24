from typing import Annotated, get_type_hints

import pytest
from pyspark.sql.types import BooleanType, LongType, StringType, StructField

from typedspark import Column, ColumnMeta, Schema
from typedspark._schema.structfield import (
    _get_structfield_dtype,
    get_structfield,
    get_structfield_meta,
)


class A(Schema):
    a: Column[LongType]
    b: Column[StringType]
    c: Annotated[Column[StringType], ColumnMeta(comment="comment")]
    d: Annotated[Column[BooleanType], ColumnMeta(comment="comment2")]


@pytest.fixture()
def type_hints():
    return get_type_hints(A, include_extras=True)


def test_get_structfield_dtype(type_hints):
    assert _get_structfield_dtype(Column[LongType], "a") == LongType()
    assert _get_structfield_dtype(type_hints["b"], "b") == StringType()
    assert (
        _get_structfield_dtype(
            Annotated[Column[StringType], ColumnMeta(comment="comment")],  # type: ignore
            "c",
        )
        == StringType()
    )
    assert _get_structfield_dtype(type_hints["d"], "d") == BooleanType()


def test_get_structfield_metadata(type_hints):
    assert get_structfield_meta(Column[LongType]) == ColumnMeta()
    assert get_structfield_meta(type_hints["b"]) == ColumnMeta()
    assert get_structfield_meta(
        Annotated[Column[StringType], ColumnMeta(comment="comment")]  # type: ignore
    ) == ColumnMeta(comment="comment")
    assert get_structfield_meta(type_hints["d"]) == ColumnMeta(comment="comment2")


def test_get_structfield(type_hints):
    assert get_structfield("a", Column[LongType]) == StructField(name="a", dataType=LongType())
    assert get_structfield("b", type_hints["b"]) == StructField(name="b", dataType=StringType())
    assert get_structfield(  # type: ignore
        "c",
        Annotated[Column[StringType], ColumnMeta(comment="comment")],  # type: ignore
    ) == StructField(name="c", dataType=StringType(), metadata={"comment": "comment"})
    assert get_structfield("d", type_hints["d"]) == StructField(
        name="d", dataType=BooleanType(), metadata={"comment": "comment2"}
    )
