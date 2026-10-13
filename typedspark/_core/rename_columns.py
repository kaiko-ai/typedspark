"""Helper functions to rename DataFrame columns between their external name (defined in
``ColumnMeta(external_name=...)``) and their internal (Python-safe) name."""

from typing import Optional, Type

from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col, lit, struct, when
from pyspark.sql.types import StructType

from typedspark._schema.schema import Schema


def rename_to_internal(df: DataFrame, schema: Type[Schema]) -> DataFrame:
    """Renames DataFrame columns from their external name
    (``ColumnMeta(external_name=...)``) to their internal name (as defined in the Schema
    class attribute)."""
    for field in schema.get_structtype().fields:
        internal_name = field.name

        if field.metadata and "external_name" in field.metadata:
            external_name = field.metadata["external_name"]
            df = df.withColumnRenamed(external_name, internal_name)

        if isinstance(field.dataType, StructType):
            structtype = _build_internal_struct(field.dataType, internal_name)
            df = df.withColumn(internal_name, structtype)

    return df


def rename_to_external(df: DataFrame, schema: Type[Schema]) -> DataFrame:
    """Renames DataFrame columns from their internal name (as defined in the Schema
    class attribute) back to their external name (``ColumnMeta(external_name=...)``)."""
    for field in schema.get_structtype().fields:
        internal_name = field.name

        if field.metadata and "external_name" in field.metadata:
            external_name = field.metadata["external_name"]
            df = df.withColumnRenamed(internal_name, external_name)

        if isinstance(field.dataType, StructType):
            col_name = (
                field.metadata.get("external_name", internal_name)
                if field.metadata
                else internal_name
            )
            structtype = _build_external_struct(field.dataType, col_name)
            df = df.withColumn(col_name, structtype)

    return df


def _build_internal_struct(
    schema: StructType,
    parent: str,
    full_parent_path: Optional[str] = None,
) -> Column:
    """Builds a Column expression that reads from external field paths and outputs a
    struct with internal field names."""
    if not full_parent_path:
        full_parent_path = f"`{parent}`"

    mapping = []
    for field in schema.fields:
        external_name = (
            field.metadata.get("external_name", field.name) if field.metadata else field.name
        )
        child_path = f"{full_parent_path}.`{external_name}`"

        if isinstance(field.dataType, StructType):
            mapping.append(
                _build_internal_struct(
                    field.dataType,
                    parent=field.name,
                    full_parent_path=child_path,
                )
            )
        else:
            mapping.append(col(child_path).alias(field.name))

    return _wrap_struct(mapping, parent, full_parent_path)


def _build_external_struct(
    schema: StructType,
    parent: str,
    full_parent_path: Optional[str] = None,
) -> Column:
    """Builds a Column expression that reads from internal field paths and outputs a
    struct with external field names."""
    if not full_parent_path:
        full_parent_path = f"`{parent}`"

    mapping = []
    for field in schema.fields:
        internal_name = field.name
        external_name = (
            field.metadata.get("external_name", internal_name) if field.metadata else internal_name
        )
        child_path = f"{full_parent_path}.`{internal_name}`"

        if isinstance(field.dataType, StructType):
            mapping.append(
                _build_external_struct(  # recurse in the same direction
                    field.dataType,
                    parent=external_name,
                    full_parent_path=child_path,
                )
            )
        else:
            mapping.append(col(child_path).alias(external_name))

    return _wrap_struct(mapping, parent, full_parent_path)


def _wrap_struct(
    mapping: list[Column],
    parent: str,
    full_parent_path: str,
) -> Column:
    """Wraps a list of column expressions in a nullable struct."""
    return (
        when(
            col(full_parent_path).isNotNull(),
            struct(*mapping),
        )
        .otherwise(lit(None))
        .alias(parent)
    )
