"""Utility functions for creating a ``Schema`` from a ``StructType``"""

from typing import Dict, Literal, Optional, Type

from pyspark.sql.types import ArrayType as SparkArrayType
from pyspark.sql.types import DataType
from pyspark.sql.types import DayTimeIntervalType as SparkDayTimeIntervalType
from pyspark.sql.types import DecimalType as SparkDecimalType
from pyspark.sql.types import MapType as SparkMapType
from pyspark.sql.types import StructType as SparkStructType

from typedspark._core.column import Column
from typedspark._core.datatypes import (
    ArrayType,
    DayTimeIntervalType,
    DecimalType,
    MapType,
    StructType,
)
from typedspark._schema.schema import MetaSchema, Schema
from typedspark._utils.camelcase import to_camel_case


def create_schema_from_structtype(
    structtype: SparkStructType, schema_name: Optional[str] = None
) -> Type[Schema]:
    """Dynamically builds a ``Schema`` based on a ``DataFrame``'s ``StructType``"""
    type_annotations = {}
    attributes: Dict[str, None] = {}
    for column in structtype:
        name = column.name
        data_type = _extract_data_type(column.dataType, name)
        type_annotations[name] = Column[data_type]  # type: ignore
        attributes[name] = None

    if not schema_name:
        schema_name = "DynamicallyLoadedSchema"

    schema = MetaSchema(schema_name, tuple([Schema]), attributes)
    schema.__annotations__ = type_annotations

    return schema  # type: ignore


def _extract_data_type(dtype: DataType, name: str) -> Type[DataType]:
    """Given an instance of a ``DataType``, it extracts the corresponding ``DataType``
    class, potentially including annotations (e.g. ``ArrayType[StringType]``)."""
    if isinstance(dtype, SparkArrayType):
        element_type = _extract_data_type(dtype.elementType, name)
        return ArrayType[element_type]  # type: ignore

    if isinstance(dtype, SparkMapType):
        key_type = _extract_data_type(dtype.keyType, name)
        value_type = _extract_data_type(dtype.valueType, name)
        return MapType[key_type, value_type]  # type: ignore

    if isinstance(dtype, SparkStructType):
        subschema = create_schema_from_structtype(dtype, to_camel_case(name))
        return StructType[subschema]  # type: ignore

    if isinstance(dtype, SparkDayTimeIntervalType):
        start_field = dtype.startField
        end_field = dtype.endField
        return DayTimeIntervalType[Literal[start_field], Literal[end_field]]  # type: ignore

    if isinstance(dtype, SparkDecimalType):
        precision = dtype.precision
        scale = dtype.scale
        return DecimalType[Literal[precision], Literal[scale]]  # type: ignore

    return type(dtype)
