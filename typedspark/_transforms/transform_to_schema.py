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


def _do_transformations(dataframe: DataFrame, transformations: Dict[str, SparkColumn]) -> DataFrame:
    """Performs the transformations on the provided DataFrame."""
    return reduce(
        lambda acc, key: DataFrame.withColumn(acc, key, transformations[key]),
        transformations.keys(),
        dataframe,
    )


def _rename_temporary_keys_to_original_keys(
    dataframe: DataFrame, problematic_key_mapping: Dict[str, str]
) -> DataFrame:
    """Renames the temporary keys back to the original keys."""
    return reduce(
        lambda acc, key: DataFrame.withColumnRenamed(acc, problematic_key_mapping[key], key),
        problematic_key_mapping.keys(),
        dataframe,
    )


def _identify_problematic_keys(
    dataframe_columns: list, transformations: Dict[str, SparkColumn]
) -> Dict[str, str]:
    """Identifies the problematic keys in the transformations dictionary."""
    problematic_keys = _duplicates(dataframe_columns) & set(transformations.keys())
    return {k: f"my_temporary_typedspark_{k}" for k in problematic_keys}


def _rename_keys_to_temporary_keys(
    transformations: Dict[str, SparkColumn], temporary_key_mapping: Dict[str, str]
) -> Dict[str, SparkColumn]:
    """Renames the keys in the transformations dictionary to temporary keys."""
    return {temporary_key_mapping.get(k, k): v for k, v in transformations.items()}


def _do_transformations(dataframe: DataFrame, transformations: Dict[str, SparkColumn]) -> DataFrame:
    """Performs the transformations on the provided DataFrame."""
    return reduce(
        lambda acc, key: DataFrame.withColumn(acc, key, transformations[key]),
        transformations.keys(),
        dataframe,
    )


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
