"""Utility functions for creating a ``Schema`` from a ``StructType``"""

import re
from typing import Annotated, Dict, Literal, Optional, Type

from pyspark.sql.types import ArrayType as SparkArrayType
from pyspark.sql.types import DataType
from pyspark.sql.types import DayTimeIntervalType as SparkDayTimeIntervalType
from pyspark.sql.types import DecimalType as SparkDecimalType
from pyspark.sql.types import MapType as SparkMapType
from pyspark.sql.types import StructType as SparkStructType

from typedspark._core.column import Column
from typedspark._core.column_meta import ColumnMeta
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
    column_name_mapping = _create_column_name_mapping(structtype)

    for column in structtype:
        name = column.name
        data_type = _extract_data_type(column.dataType, name)

        mapped_name = column_name_mapping[name]
        if mapped_name == name:
            type_annotations[name] = Column[data_type]  # type: ignore
        else:
            type_annotations[mapped_name] = Annotated[
                Column[data_type], ColumnMeta(external_name=name)
            ]

        attributes[name] = None

    if not schema_name:
        schema_name = "DynamicallyLoadedSchema"

    schema = MetaSchema(schema_name, tuple([Schema]), attributes)
    schema.__annotations__ = type_annotations

    return schema  # type: ignore


def _create_column_name_mapping(structtype: SparkStructType) -> Dict[str, str]:
    """Checks if there are duplicate columns after replacing illegal characters."""
    mapping = {column: _replace_illegal_characters(column) for column in structtype.names}

    renamed_columns = list(mapping.values())
    duplicates = {
        column: column_renamed
        for column, column_renamed in mapping.items()
        if renamed_columns.count(column_renamed) > 1
    }

    if len(duplicates) > 0:
        raise ValueError(
            "You're trying to dynamically generate a Schema from a DataFrame. "
            + "However, typedspark has detected that the DataFrame contains duplicate columns "
            + "after replacing illegal characters (e.g. whitespaces, dots, etc.).\n"
            + "The folowing columns have lead to duplicates:\n"
            + f"{duplicates}\n\n"
            + "Please rename these columns in your DataFrame."
        )

    return mapping


def _replace_illegal_characters(column_name: str) -> str:
    """Replaces illegal characters in a column name with an underscore."""
    return re.sub("[^A-Za-z0-9]", "_", column_name)


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
