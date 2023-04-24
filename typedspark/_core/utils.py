"""Utility functions for unpacking ``Column[]``, ``DataType[]``, and ``Annotated[]``."""
from typing import Annotated, Type, TypeVar, get_args, get_origin

from pyspark.sql.types import DataType

from typedspark import Column, Schema
from typedspark._core.column_meta import ColumnMeta
from typedspark._core.datatypes import TypedSparkDataType

_DataType = TypeVar("_DataType", bound=DataType)  # pylint: disable=invalid-name


def is_column(column: Type[Column[_DataType]]) -> bool:
    """Returns True if the column is of type ``Column``."""
    return get_origin(column) == Column


def is_annotated_column(column: Annotated[Type[Column[_DataType]], ColumnMeta]) -> bool:
    """Returns True if the column is of type ``Annotated[Column, ColumnMeta()]``."""
    return get_origin(column) == Annotated


def get_column_from_annotation(
    column: Annotated[Type[Column[_DataType]], ColumnMeta]
) -> Type[Column[_DataType]]:
    """Takes an ``Annotation[Column[...], ...]`` and returns the
    ``Column[...]``."""
    column = get_args(column)[0]
    if get_origin(column) != Column:
        raise TypeError("Column needs to have a Column[] within Annotated[].")

    return column


def get_dtype_from_column(column: Type[Column[_DataType]]) -> Type[DataType]:
    """Returns the DataType from a Column."""
    args = get_args(column)
    if not args:
        raise TypeError("Column needs to have a DataType argument, e.g. Column[IntegerType].")

    return args[0]


def is_of_typedspark_type(
    dtype_observed: Type[DataType], dtype_expected: Type[TypedSparkDataType]
) -> bool:
    """Returns True if the observed data type is of the expected type."""
    return get_origin(dtype_observed) == dtype_expected


def get_schema_from_structtype(dtype: Type[DataType]) -> Type[Schema]:
    """Returns the Schema from a StructType[Schema]."""
    args = get_args(dtype)
    if not args:
        raise TypeError("StructType needs to have a Schema argument, e.g. StructType[MySchema].")

    return args[0]
