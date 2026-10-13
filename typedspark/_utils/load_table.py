"""Functions for loading `DataSet` and `Schema` in notebooks."""

from typing import Optional, Tuple, Type

from pyspark.sql import DataFrame, SparkSession

from typedspark._core.dataset import DataSet
from typedspark._schema.schema import Schema
from typedspark._utils.create_dataset_from_structtype import create_schema_from_structtype


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
    schema = create_schema_from_structtype(dataframe.schema, schema_name)
    return DataSet[schema].from_dataframe(dataframe)  # type: ignore


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
