from typing import Annotated, Literal, Type

import pytest
from pyspark.sql.types import LongType, StringType, StructField, StructType

import typedspark
from typedspark import Column, ColumnMeta, Schema
from typedspark._schema.schema import DltKwargs


class A(Schema):
    a: Column[LongType]
    b: Column[StringType]


schema_a_string = """
from pyspark.sql.types import LongType, StringType

from typedspark import Column, Schema


class A(Schema):
    a: Column[LongType]
    b: Column[StringType]
"""

schema_a_string_with_documentation = '''from typing import Annotated

from pyspark.sql.types import LongType, StringType

from typedspark import Column, ColumnMeta, Schema


class A(Schema):
    """Add documentation here."""

    a: Annotated[Column[LongType], ColumnMeta(comment="")]
    b: Annotated[Column[StringType], ColumnMeta(comment="")]
'''


class B(Schema):
    b: Column[LongType]
    a: Column[StringType]


class Values(Schema):
    a: Column[typedspark.DecimalType[Literal[38], Literal[18]]]
    b: Column[StringType]


class ComplexDatatypes(Schema):
    value: Column[typedspark.StructType[Values]]
    items: Column[typedspark.ArrayType[StringType]]
    consequences: Column[typedspark.MapType[StringType, typedspark.ArrayType[StringType]]]


schema_complex_datatypes = '''from typing import Annotated, Literal

from pyspark.sql.types import StringType

from typedspark import ArrayType, Column, ColumnMeta, DecimalType, MapType, Schema, StructType


class ComplexDatatypes(Schema):
    """Add documentation here."""

    value: Annotated[Column[StructType[test_schema.Values]], ColumnMeta(comment="")]
    items: Annotated[Column[ArrayType[StringType]], ColumnMeta(comment="")]
    consequences: Annotated[Column[MapType[StringType, ArrayType[StringType]]], ColumnMeta(comment="")]


class Values(Schema):
    """Add documentation here."""

    a: Annotated[Column[DecimalType[Literal[38], Literal[18]]], ColumnMeta(comment="")]
    b: Annotated[Column[StringType], ColumnMeta(comment="")]
'''  # noqa: E501


class PascalCase(Schema):
    """Schema docstring."""

    a: Annotated[Column[StringType], ColumnMeta(comment="some")]
    b: Annotated[Column[LongType], ColumnMeta(comment="other")]


def test_all_column_names():
    assert A.all_column_names() == ["a", "b"]
    assert B.all_column_names() == ["b", "a"]


def test_all_column_names_except_for():
    assert A.all_column_names_except_for(["a"]) == ["b"]
    assert B.all_column_names_except_for([]) == ["b", "a"]
    assert B.all_column_names_except_for(["b", "a"]) == []


def test_get_snake_case():
    assert A.get_snake_case() == "a"
    assert PascalCase.get_snake_case() == "pascal_case"


def test_get_docstring():
    assert A.get_docstring() is None
    assert PascalCase.get_docstring() == "Schema docstring."


def test_get_structtype():
    assert A.get_structtype() == StructType(
        [StructField("a", LongType(), True), StructField("b", StringType(), True)]
    )
    assert PascalCase.get_structtype() == StructType(
        [
            StructField("a", StringType(), metadata={"comment": "some"}),
            StructField("b", LongType(), metadata={"comment": "other"}),
        ]
    )


def test_get_dlt_kwargs():
    assert A.get_dlt_kwargs() == DltKwargs(
        name="a",
        comment=None,
        schema=StructType(
            [StructField("a", LongType(), True), StructField("b", StringType(), True)]
        ),
    )

    assert PascalCase.get_dlt_kwargs() == DltKwargs(
        name="pascal_case",
        comment="Schema docstring.",
        schema=StructType(
            [
                StructField("a", StringType(), metadata={"comment": "some"}),
                StructField("b", LongType(), metadata={"comment": "other"}),
            ]
        ),
    )


def test_repr():
    assert repr(A) == schema_a_string


@pytest.mark.parametrize(
    "schema, expected_schema_definition",
    [
        (A, schema_a_string_with_documentation),
        (ComplexDatatypes, schema_complex_datatypes),
    ],
)
def test_get_schema(schema: Type[Schema], expected_schema_definition: str):
    schema_definition = schema.get_schema_definition_as_string(include_documentation=True)
    assert schema_definition == expected_schema_definition


def test_dtype_attributes():
    assert ComplexDatatypes.value.dtype == typedspark.StructType[Values]
    assert ComplexDatatypes.value.dtype.schema == Values
    assert ComplexDatatypes.value.dtype.schema.b.dtype == StringType()
