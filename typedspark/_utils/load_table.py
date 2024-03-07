"""Functions for loading `DataSet` and `Schema`."""

from typing import Optional, Tuple, Type

from pyspark.sql import DataFrame, SparkSession

from typedspark._core.dataset import DataSet
from typedspark._schema.schema import Schema
from typedspark._utils.create_dataset_from_structtype import create_schema_from_structtype
from typedspark._utils.register_schema_to_dataset import register_schema_to_dataset
from typedspark._utils.replace_illegal_column_names import replace_illegal_column_names


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
    dataframe = replace_illegal_column_names(dataframe)
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
