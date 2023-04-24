"""Utility functions for unpacking ``Column[]`` and ``DataType[]``."""
from __future__ import annotations

from typing import TYPE_CHECKING, Type, TypeVar, get_args, get_origin

from pyspark.sql.types import DataType

from typedspark._core.datatypes import StructType

if TYPE_CHECKING:
    from typedspark import Column, Schema

_DataType = TypeVar("_DataType", bound=DataType)  # pylint: disable=invalid-name


def get_dtype_from_column(column: Type[Column[_DataType]]) -> Type[DataType]:
    """Returns the DataType from a Column."""
    args = get_args(column)
    if not args:
        raise TypeError("Column needs to have a DataType argument, e.g. Column[IntegerType].")

    return args[0]


def is_structtype(dtype_observed: Type[DataType]) -> bool:
    """Returns True if the observed data type is typedspark's
    ``StructType``."""
    return get_origin(dtype_observed) == StructType


def get_schema_from_structtype(dtype: Type[DataType]) -> Type[Schema]:
    """Returns the Schema from a StructType[Schema]."""
    args = get_args(dtype)
    if not args:
        raise TypeError("StructType needs to have a Schema argument, e.g. StructType[MySchema].")

    return args[0]
