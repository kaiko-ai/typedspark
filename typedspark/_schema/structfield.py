"""Module responsible for generating StructFields from Columns in a Schema."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Annotated, Type, TypeVar, Union, get_args, get_origin

from pyspark.sql.types import ArrayType as SparkArrayType
from pyspark.sql.types import DataType
from pyspark.sql.types import DayTimeIntervalType as SparkDayTimeIntervalType
from pyspark.sql.types import DecimalType as SparkDecimalType
from pyspark.sql.types import MapType as SparkMapType
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType as SparkStructType

from typedspark._core.column import Column
from typedspark._core.column_meta import ColumnMeta
from typedspark._core.datatypes import (
    ArrayType,
    DayTimeIntervalType,
    DecimalType,
    MapType,
    StructType,
    TypedSparkDataType,
)

if TYPE_CHECKING:  # pragma: no cover
    from typedspark._schema.schema import Schema

_DataType = TypeVar("_DataType", bound=DataType)  # pylint: disable=invalid-name


def get_structfield(
    name: str,
    column: Union[Type[Column[_DataType]], Annotated[Type[Column[_DataType]], ColumnMeta]],
) -> StructField:
    """Generates a ``StructField`` for a given ``Column`` in a ``Schema``."""
    meta = get_structfield_meta(column)

    return StructField(
        name=name,
        dataType=_get_structfield_dtype(column, name),
        nullable=True,
        metadata=meta.get_metadata(),
    )


def get_structfield_meta(
    column: Union[Type[Column[_DataType]], Annotated[Type[Column[_DataType]], ColumnMeta]],
) -> ColumnMeta:
    """Get the spark column metadata from the ``ColumnMeta`` data, when available."""
    return next((x for x in get_args(column) if isinstance(x, ColumnMeta)), ColumnMeta())


def _get_structfield_dtype(
    column: Union[Type[Column[_DataType]], Annotated[Type[Column[_DataType]], ColumnMeta]],
    colname: str,
) -> DataType:
    """Get the spark ``DataType`` from the ``Column`` type annotation."""
    origin = get_origin(column)
    if origin not in [Annotated, Column]:
        raise TypeError(f"Column {colname} needs to be of type Column or Annotated.")

    if origin == Annotated:
        column = _get_column_from_annotation(column, colname)

    args = get_args(column)
    dtype = _get_dtype(args[0], colname)
    return dtype


def _get_column_from_annotation(
    column: Annotated[Type[Column[_DataType]], ColumnMeta],
    colname: str,
) -> Type[Column[_DataType]]:
    """Takes an ``Annotation[Column[...], ...]`` and returns the ``Column[...]``."""
    column = get_args(column)[0]
    if get_origin(column) != Column:
        raise TypeError(f"Column {colname} needs to have a Column[] within Annotated[].")

    return column


def _get_dtype(dtype: Type[DataType], colname: str) -> DataType:
    """Takes a ``DataType`` class and returns a DataType object."""
    origin = get_origin(dtype)
    if origin == ArrayType:
        return _extract_arraytype(dtype, colname)
    if origin == MapType:
        return _extract_maptype(dtype, colname)
    if origin == StructType:
        return _extract_structtype(dtype)
    if origin == DecimalType:
        return _extract_decimaltype(dtype)
    if origin == DayTimeIntervalType:
        return _extract_daytimeintervaltype(dtype)
    if (
        inspect.isclass(dtype)
        and issubclass(dtype, DataType)
        and not issubclass(dtype, TypedSparkDataType)
    ):
        return dtype()

    raise TypeError(
        f"Column {colname} does not have a correctly formatted DataType as a parameter."
    )


def _extract_arraytype(arraytype: Type[DataType], colname: str) -> SparkArrayType:
    """Takes e.g. an ``ArrayType[StringType]`` and creates an ``ArrayType(StringType(),
    True)``."""
    params = get_args(arraytype)
    element_type = _get_dtype(params[0], colname)
    return SparkArrayType(element_type)


def _extract_maptype(maptype: Type[DataType], colname: str) -> SparkMapType:
    """Takes e.g. a ``MapType[StringType, StringType]`` and creates a ``
    MapType(StringType(), StringType(), True)``."""
    params = get_args(maptype)
    key_type = _get_dtype(params[0], colname)
    value_type = _get_dtype(params[1], colname)
    return SparkMapType(key_type, value_type)


def _extract_structtype(structtype: Type[DataType]) -> SparkStructType:
    """Takes a ``StructType[Schema]`` annotation and creates a
    ``StructType(schema_list)``, where ``schema_list`` contains all ``StructField()``
    defined in the ``Schema``."""
    params = get_args(structtype)
    schema: Type[Schema] = params[0]
    return schema.get_structtype()


def _extract_decimaltype(decimaltype: Type[DataType]) -> SparkDecimalType:
    """Takes e.g. a ``DecimalType[Literal[10], Literal[12]]`` and returns
    ``DecimalType(10, 12)``."""
    params = get_args(decimaltype)
    key_type: int = _unpack_literal(params[0])
    value_type: int = _unpack_literal(params[1])
    return SparkDecimalType(key_type, value_type)


def _extract_daytimeintervaltype(daytimeintervaltype: Type[DataType]) -> SparkDayTimeIntervalType:
    """Takes e.g. a ``DayTimeIntervalType[Literal[1], Literal[2]]`` and returns
    ``DayTimeIntervalType(1, 2)``."""
    params = get_args(daytimeintervaltype)
    start_field: int = _unpack_literal(params[0])
    end_field: int = _unpack_literal(params[1])
    return SparkDayTimeIntervalType(start_field, end_field)


def _unpack_literal(literal):
    """Takes as input e.g. ``Literal[10]`` and returns ``10``."""
    return get_args(literal)[0]
