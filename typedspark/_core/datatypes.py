"""Here, we make our own definitions of ``MapType``, ``ArrayType`` and ``StructType`` in
order to allow e.g. for ``ArrayType[StringType]``."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Dict, Generic, Type, TypeVar, get_args, get_origin

from pyspark.sql.types import ArrayType as SparkArrayType
from pyspark.sql.types import DataType
from pyspark.sql.types import DayTimeIntervalType as SparkDayTimeIntervalType
from pyspark.sql.types import DecimalType as SparkDecimalType
from pyspark.sql.types import MapType as SparkMapType
from pyspark.sql.types import StructType as SparkStructType

if TYPE_CHECKING:  # pragma: no cover
    from typedspark._core.column import Column
    from typedspark._schema.schema import Schema

    _Schema = TypeVar("_Schema", bound=Schema)
else:
    _Schema = TypeVar("_Schema")

_KeyType = TypeVar("_KeyType", bound=DataType)  # pylint: disable=invalid-name
_ValueType = TypeVar("_ValueType", bound=DataType)  # pylint: disable=invalid-name
_Precision = TypeVar("_Precision", bound=int)  # pylint: disable=invalid-name
_Scale = TypeVar("_Scale", bound=int)  # pylint: disable=invalid-name
_StartField = TypeVar("_StartField", bound=int)  # pylint: disable=invalid-name
_EndField = TypeVar("_EndField", bound=int)  # pylint: disable=invalid-name


class TypedSparkDataType(DataType):
    """Base class for typedspark specific ``DataTypes``."""

    @classmethod
    def get_name(cls) -> str:
        """Return the name of the type."""
        return cls.__name__


class StructTypeMeta(type):
    """Initializes the schema attribute as None.

    This allows for auto-complete in Databricks notebooks (uninitialized variables don't
    show up in auto-complete there).
    """

    def __new__(cls, name: str, bases: Any, dct: Dict[str, Any]):
        dct["schema"] = None
        return super().__new__(cls, name, bases, dct)


class StructType(Generic[_Schema], TypedSparkDataType, metaclass=StructTypeMeta):
    """Allows for type annotations such as:

    .. code-block:: python

        class Job(Schema):
            position: Column[StringType]
            salary: Column[LongType]

        class Person(Schema):
            job: Column[StructType[Job]]
    """

    def __init__(
        self,
        schema: Type[_Schema],
        parent: Column[Any],
    ) -> None:
        self.schema = schema
        self.schema._parent = parent


class MapType(Generic[_KeyType, _ValueType], TypedSparkDataType):
    """Allows for type annotations such as.

    .. code-block:: python

        class Basket(Schema):
            items: Column[MapType[StringType, StringType]]
    """


class ArrayType(Generic[_ValueType], TypedSparkDataType):
    """Allows for type annotations such as.

    .. code-block:: python

        class Basket(Schema):
            items: Column[ArrayType[StringType]]
    """


class DecimalType(Generic[_Precision, _Scale], TypedSparkDataType):
    """Allows for type annotations such as.

    .. code-block:: python

        class Numbers(Schema):
            number: Column[DecimalType[Literal[10], Literal[0]]]
    """


class DayTimeIntervalType(Generic[_StartField, _EndField], TypedSparkDataType):
    """Allows for type annotations such as.

    .. code-block:: python

        class TimeInterval(Schema):
            interval: Column[DayTimeIntervalType[IntervalType.HOUR, IntervalType.SECOND]
    """


def materialize_dtype(dtype: Type[DataType], colname: str) -> DataType:
    """Create a PySpark ``DataType`` from a TypedSpark type annotation."""
    origin = get_origin(dtype)
    if origin == ArrayType:
        params = get_args(dtype)
        return SparkArrayType(materialize_dtype(params[0], colname))
    if origin == MapType:
        params = get_args(dtype)
        return SparkMapType(
            materialize_dtype(params[0], colname),
            materialize_dtype(params[1], colname),
        )
    if origin == StructType:
        schema: Type[Schema] = get_args(dtype)[0]
        return schema.get_structtype()
    if origin == DecimalType:
        precision, scale = get_args(dtype)
        return SparkDecimalType(_unpack_literal(precision), _unpack_literal(scale))
    if origin == DayTimeIntervalType:
        start_field, end_field = get_args(dtype)
        return SparkDayTimeIntervalType(
            _unpack_literal(start_field),
            _unpack_literal(end_field),
        )
    if (
        inspect.isclass(dtype)
        and issubclass(dtype, DataType)
        and not issubclass(dtype, TypedSparkDataType)
    ):
        return dtype()

    raise TypeError(
        f"Column {colname} does not have a correctly formatted DataType as a parameter."
    )


def _unpack_literal(literal):
    """Return the value wrapped by a ``Literal`` annotation."""
    return get_args(literal)[0]
