"""Here, we make our own definitions of `MapType`, `ArrayType` and `StructType`
in order to allow e.g. for `ArrayType[StringType]`."""
from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Generic, TypeVar

from pyspark.sql.types import DataType

if TYPE_CHECKING:  # pragma: no cover
    from typedspark._schema.schema import Schema

    _Schema = TypeVar("_Schema", bound=Schema)
else:
    _Schema = TypeVar("_Schema")

_KeyType = TypeVar("_KeyType", bound=DataType)  # pylint: disable=invalid-name
_ValueType = TypeVar("_ValueType", bound=DataType)  # pylint: disable=invalid-name
_Precision = TypeVar("_Precision", bound=int)  # pylint: disable=invalid-name
_Scale = TypeVar("_Scale", bound=int)  # pylint: disable=invalid-name


class TypedSparkDataType(DataType, ABC):
    """Abstract base class for typedspark specific `DataTypes`."""


class StructType(Generic[_Schema], TypedSparkDataType):
    """Allows for type annotations such as:"""


class MapType(Generic[_KeyType, _ValueType], TypedSparkDataType):
    """Allows for type annotations such as `Column[MapType[StringType,
    StringType]]`"""


class ArrayType(Generic[_ValueType], TypedSparkDataType):
    """Allows for type annotations such as `Column[ArrayType[StringType]]"""


class DecimalType(Generic[_Precision, _Scale], TypedSparkDataType):
    """Allows for type annotations such as `Column[DecimalType[Literal[10],
    Literal[0]]]."""
