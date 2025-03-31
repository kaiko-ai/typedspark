"""Module containing functions that are related to transformations to DataSets."""

from functools import reduce
from typing import Dict, Optional, Type, TypeVar, Union

from pyspark.sql import Column as SparkColumn
from pyspark.sql import DataFrame

from typedspark._core.column import Column
from typedspark._core.dataset import DataSet
from typedspark._schema.schema import Schema
from typedspark._transforms.rename_duplicate_columns import RenameDuplicateColumns
from typedspark._transforms.utils import add_nulls_for_unspecified_columns, convert_keys_to_strings

T = TypeVar("T", bound=Schema)


def _do_transformations(
    dataframe: DataFrame, transformations: Dict[str, SparkColumn], run_sequentially: bool = True
) -> DataFrame:
    """Performs the transformations on the provided DataFrame."""
    if run_sequentially:
        return reduce(
            lambda acc, key: DataFrame.withColumn(acc, key, transformations[key]),
            transformations.keys(),
            dataframe,
        )
    return DataFrame.withColumns(dataframe, transformations)


def _rename_temporary_keys_to_original_keys(
    dataframe: DataFrame, problematic_key_mapping: Dict[str, str]
) -> DataFrame:
    """Renames the temporary keys back to the original keys."""
    return reduce(
        lambda acc, key: DataFrame.withColumnRenamed(acc, problematic_key_mapping[key], key),
        problematic_key_mapping.keys(),
        dataframe,
    )


def transform_to_schema(
    dataframe: DataFrame,
    schema: Type[T],
    transformations: Optional[Dict[Column, SparkColumn]] = None,
    fill_unspecified_columns_with_nulls: bool = False,
    run_sequentially: bool = True,
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
    transform: Union[dict[str, SparkColumn], RenameDuplicateColumns]
    transform = convert_keys_to_strings(transformations)

    if fill_unspecified_columns_with_nulls:
        transform = add_nulls_for_unspecified_columns(transform, schema, dataframe.columns)

    transform = RenameDuplicateColumns(transform, schema, dataframe.columns)

    return DataSet[schema](  # type: ignore
        dataframe.transform(_do_transformations, transform.transformations, run_sequentially)
        .drop(*transform.temporary_key_mapping.keys())
        .transform(_rename_temporary_keys_to_original_keys, transform.temporary_key_mapping)
        .select(*schema.all_column_names())
    )
