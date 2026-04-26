"""Util functions for typedspark._transforms."""

from typing import Any, Dict, List, Optional, Type

from pyspark.sql import Column as SparkColumn
from pyspark.sql.functions import col, lit, transform, transform_keys, transform_values
from pyspark.sql.types import ArrayType, MapType, StructType

from typedspark._core.column import Column
from typedspark._core.validate_schema import unpack_schema
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
    transformations: Optional[Dict[Column[Any], SparkColumn]],
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


def _fill_missing_fields(
    schema_dtype,
    data_dtype,
    current: SparkColumn,
) -> Optional[SparkColumn]:
    """Return `current` rebuilt with any schema fields absent from the data added as null.
    Works on column expressions (not dotted-path strings) so it can recurse into array
    and map elements via higher-order function lambdas. Returns None when no changes are
    needed.
    """
    if schema_dtype == data_dtype:
        return None
    if isinstance(schema_dtype, StructType) and isinstance(data_dtype, StructType):
        data_dict = unpack_schema(data_dtype)
        expression = current
        changed = False
        for schema_field in schema_dtype.fields:
            if schema_field.name not in data_dict:
                expression = expression.withField(
                    schema_field.name, lit(None).cast(schema_field.dataType)
                )
                changed = True
                continue
            inner = _fill_missing_fields(
                schema_field.dataType,
                data_dict[schema_field.name].dataType,
                current.getField(schema_field.name),
            )
            if inner is not None:
                expression = expression.withField(schema_field.name, inner)
                changed = True
        return expression if changed else None
    if isinstance(schema_dtype, ArrayType) and isinstance(data_dtype, ArrayType):
        element_schema = schema_dtype.elementType
        element_data = data_dtype.elementType
        if _fill_missing_fields(element_schema, element_data, col("_")) is None:
            return None
        return transform(
            current,
            lambda elem: _fill_missing_fields_or_keep(element_schema, element_data, elem),
        )
    if isinstance(schema_dtype, MapType) and isinstance(data_dtype, MapType):
        key_schema = schema_dtype.keyType
        key_data = data_dtype.keyType
        value_schema = schema_dtype.valueType
        value_data = data_dtype.valueType
        expression = current
        if key_schema != key_data:
            expression = transform_keys(
                expression,
                lambda k, _v: _fill_missing_fields_or_keep(key_schema, key_data, k),
            )
        if value_schema != value_data:
            expression = transform_values(
                expression,
                lambda _k, v: _fill_missing_fields_or_keep(value_schema, value_data, v),
            )
        return expression
    return None


def _fill_missing_fields_or_keep(schema_dtype, data_dtype, current: SparkColumn) -> SparkColumn:
    filled = _fill_missing_fields(schema_dtype, data_dtype, current)
    return current if filled is None else filled


def add_nulls_for_unspecified_nested_fields(
    schema_struct: StructType,
    data_struct: StructType,
    transformations: Dict[str, SparkColumn] | None = None,
) -> Dict[str, SparkColumn]:
    """
    For each top-level column present in both `schema_struct` and `data_struct` whose
    type differs, produces a replacement expression that keeps all existing values and
    adds any schema fields missing from the data as null. Handles structs, arrays of
    structs, and maps of structs, recursively.
    """
    schema_dict = unpack_schema(schema_struct)
    data_dict = unpack_schema(data_struct)
    transformed_col_names = transformations or dict()
    result: Dict[str, SparkColumn] = dict()
    for _, schema_field in schema_dict.items():
        if schema_field.name in transformed_col_names:
            continue
        if schema_field.name not in data_dict:
            continue
        patched = _fill_missing_fields(
            schema_field.dataType,
            data_dict[schema_field.name].dataType,
            col(schema_field.name),
        )
        if patched is not None:
            result[schema_field.name] = patched
    return result
