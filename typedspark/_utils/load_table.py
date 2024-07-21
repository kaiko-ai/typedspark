"""Functions for loading `DataSet` and `Schema` in notebooks."""

import re
from typing import Dict, Optional, Tuple, Type

from pyspark.sql import DataFrame, SparkSession

from typedspark._core.dataset import DataSet
from typedspark._schema.schema import Schema
from typedspark._utils.create_dataset_from_structtype import create_schema_from_structtype
from typedspark._utils.register_schema_to_dataset import register_schema_to_dataset


def _replace_illegal_column_names(dataframe: DataFrame) -> DataFrame:
    """Replaces illegal column names with a legal version."""
    mapping = _create_mapping(dataframe)

    for column, column_renamed in mapping.items():
        if column != column_renamed:
            dataframe = dataframe.withColumnRenamed(column, column_renamed)

    return dataframe


def _create_mapping(dataframe: DataFrame) -> Dict[str, str]:
    """Checks if there are duplicate columns after replacing illegal characters."""
    mapping = {column: _replace_illegal_characters(column) for column in dataframe.columns}
    renamed_columns = list(mapping.values())
    duplicates = {
        column: column_renamed
        for column, column_renamed in mapping.items()
        if renamed_columns.count(column_renamed) > 1
    }

    if len(duplicates) > 0:
        raise ValueError(
            "You're trying to dynamically generate a Schema from a DataFrame. "
            + "However, typedspark has detected that the DataFrame contains duplicate columns "
            + "after replacing illegal characters (e.g. whitespaces, dots, etc.).\n"
            + "The folowing columns have lead to duplicates:\n"
            + f"{duplicates}\n\n"
            + "Please rename these columns in your DataFrame."
        )

    return mapping


def _replace_illegal_characters(column_name: str) -> str:
    """Replaces illegal characters in a column name with an underscore."""
    return re.sub("[^A-Za-z0-9]", "_", column_name)


def create_schema(
    dataframe: DataFrame, schema_name: Optional[str] = None
) -> Tuple[DataSet[Schema], Type[Schema]]:
    """This function inferres a ``Schema`` in a notebook based on a the provided
    ``DataFrame``.

    This allows for autocompletion on column names, amongst other
    things.

    .. code-block:: python

        df, Person = create_schema(df)
    """
    dataframe = _replace_illegal_column_names(dataframe)
    schema = create_schema_from_structtype(dataframe.schema, schema_name)
    dataset = DataSet[schema](dataframe)  # type: ignore
    schema = register_schema_to_dataset(dataset, schema)
    return dataset, schema


def load_table(
    spark: SparkSession, table_name: str, schema_name: Optional[str] = None
) -> Tuple[DataSet[Schema], Type[Schema]]:
    """This function loads a ``DataSet``, along with its inferred ``Schema``, in a
    notebook.

    This allows for autocompletion on column names, amongst other
    things.

    .. code-block:: python

        df, Person = load_table(spark, "path.to.table")
    """
    dataframe = spark.table(table_name)
    return create_schema(dataframe, schema_name)


DataFrame.to_typedspark = create_schema  # type: ignore
