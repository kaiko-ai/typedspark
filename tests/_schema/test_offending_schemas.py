from typing import Annotated, List, Type

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType

from typedspark import ArrayType, Column, ColumnMeta, MapType, Schema, create_empty_dataset
from typedspark._core.datatypes import DecimalType


class InvalidColumn(Schema):
    a: int


class ColumnWithoutType(Schema):
    a: Column


class AnnotationWithoutColumn(Schema):
    a: Annotated  # type: ignore


class InvalidColumnMeta(Schema):
    a: Annotated[StringType, str]


class InvalidDataTypeWithinAnnotation(Schema):
    a: Annotated[str, ColumnMeta()]  # type: ignore


class InvalidDataType(Schema):
    a: Column[int]  # type: ignore


class ComplexTypeWithoutSubtype(Schema):
    a: Column[ArrayType]


class ComplexTypeWithInvalidSubtype(Schema):
    a: Column[ArrayType[int]]  # type: ignore


class InvalidDataTypeWithArguments(Schema):
    a: Column[List[str]]  # type: ignore


class DecimalTypeWithoutArguments(Schema):
    a: Column[DecimalType]  # type: ignore


class DecimalTypeWithIncorrectArguments(Schema):
    a: Column[DecimalType[int, int]]  # type: ignore


offending_schemas: List[Type[Schema]] = [
    InvalidColumn,
    ColumnWithoutType,
    AnnotationWithoutColumn,
    InvalidColumnMeta,
    InvalidDataTypeWithinAnnotation,
    InvalidDataType,
    ComplexTypeWithoutSubtype,
    ComplexTypeWithInvalidSubtype,
    InvalidDataTypeWithArguments,
]


def test_offending_schema_exceptions(spark: SparkSession):
    for schema in offending_schemas:
        with pytest.raises(TypeError):
            create_empty_dataset(spark, schema)


def test_offending_schemas_repr_exceptions():
    for schema in offending_schemas:
        schema.get_schema_definition_as_string(generate_imports=True)


def test_offending_schemas_dtype():
    with pytest.raises(TypeError):
        ColumnWithoutType.a.dtype


def test_offending_schemas_runtime_error_on_load():
    with pytest.raises(TypeError):

        class WrongNumberOfArguments(Schema):
            a: Column[MapType[StringType]]  # type: ignore
