"""Util functions for typedspark._transforms."""

from typing import Dict, List, Optional, Type

from pyspark.sql import Column as SparkColumn
from pyspark.sql.functions import lit

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
