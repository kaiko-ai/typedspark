from typing import Annotated
from pyspark.sql import Column
from pyspark.sql.types import IntegerType, StringType
from typedspark._core.column_meta import ColumnMeta
from typedspark._core.datatypes import DayTimeIntervalType
from typedspark._core.literaltype import IntervalType
from typedspark._schema.get_schema_definition import _replace_literal, _replace_literals
from typedspark._schema.schema import Schema


class A(Schema):
    """"This is a docstring for A."""
    a: Annotated[Column[IntegerType], "Some column"]
    b: Column[StringType]


def test_replace_literal():
    result = _replace_literal(
        "DayTimeIntervalType[Literal[0], Literal[1]]",
        replace_literals_in=DayTimeIntervalType,
        original="Literal[0]",
        replacement="IntervalType.DAY",
    )
    expected = "DayTimeIntervalType[IntervalType.DAY, Literal[1]]"

    assert result == expected


def test_replace_literals():
    result = _replace_literals(
        "DayTimeIntervalType[Literal[0], Literal[1]]",
        replace_literals_in=DayTimeIntervalType,
        replace_literals_by=IntervalType,
    )
    expected = "DayTimeIntervalType[IntervalType.DAY, IntervalType.HOUR]"

    assert result == expected


def test_get_schema_definition_as_string(
        # schema: Type[Schema],
        # include_documentation: bool,
        # generate_imports: bool,
        # add_subschemas: bool,
        # class_name: str = "MyNewSchema",
    ):
    result = A.get_schema_definition_as_string(include_documentation=True)
    expected = ('''
    from pyspark.sql.types import IntegerType, StringType

    from typedspark import Column, Schema


    class A(Schema):
    """This is a docstring for A."""
        a: Annotated[Column[IntegerType], ColumnMeta(comment="Some column")]
        b: Annotated[Column[StringType], ColumnMeta(comment="")]"'
    ''')
    assert result == expected
