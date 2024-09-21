"""Helper functions to rename columns from their external name (defined in
`ColumnMeta(external_name=...)`) to their internal name."""

from typing import Optional

from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col, lit, struct, when
from pyspark.sql.types import StructField, StructType


def rename_columns(df: DataFrame, schema: StructType) -> DataFrame:
    """Helper functions to rename columns from their external name (defined in
    `ColumnMeta(external_name=...)`) to their internal name."""
    for field in schema.fields:
        internal_name = field.name

        if field.metadata and "external_name" in field.metadata:
            external_name = field.metadata["external_name"]
            df = df.withColumnRenamed(external_name, internal_name)

        if isinstance(field.dataType, StructType):
            structtype = _create_renamed_structtype(field.dataType, internal_name)
            df = df.withColumn(internal_name, structtype)

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
        external_name = _get_external_name(field, full_parent_path)

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


def _get_external_name(field: StructField, full_parent_path: str) -> str:
    external_name = field.metadata.get("external_name", field.name)
    return f"{full_parent_path}.`{external_name}`"


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
