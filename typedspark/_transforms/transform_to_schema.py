"""Module containing functions that are related to transformations to DataSets."""
from functools import reduce
from typing import Dict, Optional, Type, TypeVar

from pyspark.sql import Column as SparkColumn
from pyspark.sql import DataFrame

from typedspark._core.column import Column
from typedspark._core.dataset import DataSet
from typedspark._schema.schema import Schema
from typedspark._transforms.utils import add_nulls_for_unspecified_columns, convert_keys_to_strings

T = TypeVar("T", bound=Schema)


def _duplicates(lst: list) -> set:
    return {x for x in lst if lst.count(x) > 1}


def transform_to_schema(
    dataframe: DataFrame,
    schema: Type[T],
    transformations: Optional[Dict[Column, SparkColumn]] = None,
    fill_unspecified_columns_with_nulls: bool = False,
) -> DataSet[T]:
    """On the provided DataFrame ``df``, it performs the ``transformations`` (if
    provided), and subsequently subsets the resulting DataFrame to the columns specified
    in ``schema``.

    .. code-block:: python

        transform_to_schema(
            df_a.join(df_b, A.a == B.f),
            AB,
            {
                AB.a: A.a + 3,
                AB.b: A.b + 7,
                AB.i: B.i - 5,
                AB.j: B.j + 1,
            }
        )
    """
    _transformations = convert_keys_to_strings(transformations)

    if fill_unspecified_columns_with_nulls:
        _transformations = add_nulls_for_unspecified_columns(
            _transformations, schema, previously_existing_columns=dataframe.columns
        )

    problematic_keys = _duplicates(dataframe.columns) & set(_transformations.keys())
    problematic_key_mapping = {k: f"my_temporary_typedspark_{k}" for k in problematic_keys}
    _transformations = {problematic_key_mapping.get(k, k): v for k, v in _transformations.items()}

    return DataSet[schema](  # type: ignore
        reduce(
            lambda acc, key: DataFrame.withColumn(acc, key, _transformations[key]),
            _transformations.keys(),
            dataframe,
        )
        .drop(*problematic_key_mapping.keys())
        .transform(
            lambda df: reduce(
                lambda acc, key: DataFrame.withColumnRenamed(
                    acc, problematic_key_mapping[key], key
                ),
                problematic_key_mapping.keys(),
                df,
            )
        )
        .select(*schema.all_column_names())
    )
