"""Module containing functions related to creating a DataSet from scratch."""
from typing import Any, Dict, List, Type, TypeVar, get_type_hints

from pyspark.sql import SparkSession

from typedspark._core.column import Column
from typedspark._core.dataset import DataSet
from typedspark._schema.schema import Schema

T = TypeVar("T", bound=Schema)


def create_empty_dataset(spark: SparkSession, schema: Type[T], n_rows: int = 3) -> DataSet[T]:
    """Creates a `DataSet` with `Schema` schema, containing `n_rows` rows,
    filled with None values."""
    n_cols = len(get_type_hints(schema))
    rows = tuple([None] * n_cols)
    data = [rows] * n_rows
    spark_schema = schema.get_structtype()
    dataframe = spark.createDataFrame(data, spark_schema)
    return DataSet[schema](dataframe)  # type: ignore


def create_partially_filled_dataset(
    spark: SparkSession, schema: Type[T], data: Dict[Column, List[Any]]
) -> DataSet[T]:
    """Creates a `DataSet` with `Schema` schema, where `data` is a mapping from
    column to data in the respective column.

    Any columns in the schema that are not present in the data will be
    initialized with None values.
    """
    data_converted = {k.str: v for k, v in data.items()}
    n_rows_unique = {len(v) for _, v in data.items()}
    if len(n_rows_unique) > 1:
        raise ValueError("The number of rows in the provided data differs per column.")

    n_rows = list(n_rows_unique)[0]
    col_data = []
    for col in get_type_hints(schema).keys():
        if col in data_converted:
            col_data += [data_converted[col]]
        else:
            col_data += [[None] * n_rows]

    row_data = zip(*col_data)
    spark_schema = schema.get_structtype()
    dataframe = spark.createDataFrame(row_data, spark_schema)
    return DataSet[schema](dataframe)  # type: ignore
