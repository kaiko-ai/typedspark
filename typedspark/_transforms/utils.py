"""Util functions for typedspark._transforms."""

from typing import Dict, List, Optional, Type

from pyspark.sql import Column as SparkColumn
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType

from typedspark import structtype_column
from typedspark._core.column import Column
from typedspark._schema.schema import Schema


def add_nulls_for_unspecified_columns(
    transformations: Dict[str, SparkColumn],
    schema: Type[Schema],
    previously_existing_columns: Optional[List[str]] = None,
) -> Dict[str, SparkColumn]:
    """Takes the columns from the schema that are not present in the transformation
    dictionary and sets their values to Null (casted to the corresponding type defined
    in the schema)."""
    _previously_existing_columns = (
        [] if previously_existing_columns is None else previously_existing_columns
    )
    for field in schema.get_structtype().fields:
        if field.name not in transformations and field.name not in _previously_existing_columns:
            transformations[field.name] = lit(None).cast(field.dataType)

    return transformations


def convert_keys_to_strings(
    transformations: Optional[Dict[Column, SparkColumn]],
) -> Dict[str, SparkColumn]:
    """Takes the Column keys in transformations and converts them to strings."""
    if transformations is None:
        return {}

    _transformations = {k.str: v for k, v in transformations.items()}

    if len(transformations) != len(_transformations):
        raise ValueError(
            "The transformations dictionary requires columns with unique names as keys. "
            + "It is currently not possible to have ambiguous column names here, "
            + "even when used in combination with register_schema_to_dataset()."
        )

    return _transformations

def _transform_struct_column_fill_nulls(
        ts_schema: Schema,
        data_schema : StructType,
        struct_column: Column,
):
    return structtype_column(
        ts_schema,
        {
            getattr(ts_schema, field_name): (
                _transform_struct_column_fill_nulls(
                    getattr(ts_schema, field_name).dtype,
                    data_schema[field_name].dataType,
                    getattr(struct_column, field_name)
                )
                if data_schema[field_name].dataType == StructType
                else
                getattr(struct_column, field_name)
            )
            for field_name in data_schema.fieldNames()
        },
        fill_unspecified_columns_with_nulls=True,
    )

def add_nulls_for_unspecified_nested_fields(
    transformations: Dict[str, SparkColumn],
    schema: Type[Schema],
    data_schema: StructType,
) -> Dict[str, SparkColumn]:
    """Takes the columns from the target schema that are structs and not present in the
    transformations and validates that all their fields are in the source data, if they are not
    then fill them with nulls)."""
    for field in schema.get_structtype().fields:
        if (
                field.name not in transformations and
                field.dataType == StructType and
                field.name in data_schema.fieldNames):
            transformations[field.name] = _transform_struct_column_fill_nulls(
                schema, data_schema, getattr(schema, field.name)
            )

    return transformations
