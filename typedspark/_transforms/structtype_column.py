"""Functionality for dealing with StructType columns."""

from typing import Dict, Optional, Type

from pyspark.sql import Column as SparkColumn
from pyspark.sql.functions import struct

from typedspark._core.column import Column
from typedspark._schema.schema import Schema
from typedspark._transforms.utils import add_nulls_for_unspecified_columns, convert_keys_to_strings


def structtype_column(
    schema: Type[Schema],
    transformations: Optional[Dict[Column, SparkColumn]] = None,
    fill_unspecified_columns_with_nulls: bool = False,
) -> SparkColumn:
    """Helps with creating new ``StructType`` columns of a certain schema, for
    example:

    .. code-block:: python

        transform_to_schema(
            df,
            Output,
            {
                Output.values: structtype_column(
                    Value,
                    {
                        Value.a: Input.a + 2,
                        ...
                    }
                )
            }
        )
    """
    _transformations = convert_keys_to_strings(transformations)

    if fill_unspecified_columns_with_nulls:
        _transformations = add_nulls_for_unspecified_columns(_transformations, schema)

    _transformations = _order_columns(_transformations, schema)

    return struct([v.alias(k) for k, v in _transformations.items()])


def _order_columns(
    transformations: Dict[str, SparkColumn], schema: Type[Schema]
) -> Dict[str, SparkColumn]:
    """Chispa's DataFrame comparer doesn't deal nicely with StructTypes whose columns
    are ordered differently, hence we order them the same as in the schema here."""
    transformations_ordered = {}
    for field in schema.get_structtype().fields:
        transformations_ordered[field.name] = transformations[field.name]

    return transformations_ordered
