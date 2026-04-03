"""Helper functions to rename columns from their external name (defined in
`ColumnMeta(external_name=...)`) to their internal name."""

from typing import Optional, Type

from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col, lit, struct, when
from pyspark.sql.types import StructField, StructType

from typedspark._schema.schema import Schema


def rename_columns(df: DataFrame, schema: Type[Schema]) -> DataFrame:
    """Helper functions to rename columns from their external name (defined in
    `ColumnMeta(external_name=...)`) to their internal name (as used in the Schema)."""
    for field in schema.get_structtype().fields:
        internal_name = field.name

        if field.metadata and "external_name" in field.metadata:
            external_name = field.metadata["external_name"]
            df = df.withColumnRenamed(external_name, internal_name)

        if isinstance(field.dataType, StructType):
            structtype = _create_renamed_structtype(field.dataType, internal_name)
            df = df.withColumn(internal_name, structtype)

    return df


def rename_columns_2(df: DataFrame, schema: Type[Schema]) -> DataFrame:
    """Helper functions to rename columns from their internal name (as used in the
    Schema) to their external name (defined in `ColumnMeta(external_name=...)`)."""
    for field in schema.get_structtype().fields:
        internal_name = field.name

        if field.metadata and "external_name" in field.metadata:
            external_name = field.metadata["external_name"]
            df = df.withColumnRenamed(internal_name, external_name)  # swap

        if isinstance(field.dataType, StructType):
            structtype = _create_renamed_structtype_2(field.dataType, internal_name)
            df = df.withColumn(external_name, structtype)  # swap

    return df


def _create_renamed_structtype(
    schema: StructType,
    parent: str,
    full_parent_path: Optional[str] = None,
) -> Column:
    if not full_parent_path:
        full_parent_path = f"`{parent}`"

    mapping = []
    for field in schema.fields:
        external_name = _get_updated_parent_path(full_parent_path, field)

        if isinstance(field.dataType, StructType):
            mapping += [
                _create_renamed_structtype(
                    field.dataType,
                    parent=field.name,
                    full_parent_path=external_name,
                )
            ]
        else:
            mapping += [col(external_name).alias(field.name)]

    return _produce_nested_structtype(mapping, parent, full_parent_path)


def _create_renamed_structtype_2(
    schema: StructType,
    parent: str,
    full_parent_path: Optional[str] = None,
) -> Column:
    if not full_parent_path:
        full_parent_path = f"`{parent}`"

    mapping = []
    for field in schema.fields:
        internal_name = field.name
        external_name = field.metadata.get("external_name", internal_name)

        updated_parent_path = _get_updated_parent_path_2(full_parent_path, internal_name)  # swap

        if isinstance(field.dataType, StructType):
            mapping += [
                _create_renamed_structtype(
                    field.dataType,
                    parent=external_name,  # swap
                    full_parent_path=updated_parent_path,
                )
            ]
        else:
            mapping += [col(updated_parent_path).alias(external_name)]  # swap

    return _produce_nested_structtype(mapping, parent, full_parent_path)


def _get_updated_parent_path(full_parent_path: str, field: StructField) -> str:
    external_name = field.metadata.get("external_name", field.name)
    return f"{full_parent_path}.`{external_name}`"


def _get_updated_parent_path_2(full_parent_path: str, field: str) -> str:
    return f"{full_parent_path}.`{field}`"


def _produce_nested_structtype(
    mapping: list[Column],
    parent: str,
    full_parent_path: str,
) -> Column:
    return (
        when(
            col(full_parent_path).isNotNull(),
            struct(*mapping),
        )
        .otherwise(lit(None))
        .alias(parent)
    )
