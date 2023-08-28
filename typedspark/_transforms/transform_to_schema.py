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
    """Returns a set of the duplicates in the provided list."""
    return {x for x in lst if lst.count(x) > 1}


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
    _transformations = convert_keys_to_strings(transformations)

    if fill_unspecified_columns_with_nulls:
        _transformations = add_nulls_for_unspecified_columns(
            _transformations, schema, previously_existing_columns=dataframe.columns
        )

    temporary_key_mapping = _identify_problematic_keys(dataframe.columns, _transformations)
    _transformations = _rename_keys_to_temporary_keys(_transformations, temporary_key_mapping)

    return DataSet[schema](  # type: ignore
        dataframe.transform(_do_transformations, _transformations)
        .drop(*temporary_key_mapping.keys())
        .transform(_rename_temporary_keys_to_original_keys, temporary_key_mapping)
        .select(*schema.all_column_names())
    )
