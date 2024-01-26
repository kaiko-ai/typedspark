"""Module containing functions related to creating a DataSet from scratch."""

from typing import Any, Dict, List, Type, TypeVar, Union, get_type_hints

from pyspark.sql import Row, SparkSession

from typedspark._core.column import Column
from typedspark._core.dataset import DataSet
from typedspark._schema.schema import Schema

T = TypeVar("T", bound=Schema)


def create_empty_dataset(spark: SparkSession, schema: Type[T], n_rows: int = 3) -> DataSet[T]:
    """Creates a ``DataSet`` with ``Schema`` schema, containing ``n_rows`` rows, filled
    with ``None`` values.

    .. code-block:: python

        class Person(Schema):
            name: Column[StringType]
            age: Column[LongType]

        df = create_empty_dataset(spark, Person)
    """
    n_cols = len(get_type_hints(schema))
    rows = tuple([None] * n_cols)
    data = [rows] * n_rows
    spark_schema = schema.get_structtype()
    dataframe = spark.createDataFrame(data, spark_schema)
    return DataSet[schema](dataframe)  # type: ignore


def create_partially_filled_dataset(
    spark: SparkSession,
    schema: Type[T],
    data: Union[Dict[Column, List[Any]], List[Dict[Column, Any]]],
) -> DataSet[T]:
    """Creates a ``DataSet`` with ``Schema`` schema, where ``data`` can
    be defined in either of the following two ways:

    .. code-block:: python

        class Person(Schema):
            name: Column[StringType]
            age: Column[LongType]
            job: Column[StringType]

        df = create_partially_filled_dataset(
            spark,
            Person,
            {
                Person.name: ["John", "Jack", "Jane"],
                Person.age: [30, 40, 50],
            }
        )

    Or:

    .. code-block:: python

        df = create_partially_filled_dataset(
            spark,
            Person,
            [
                {Person.name: "John", Person.age: 30},
                {Person.name: "Jack", Person.age: 40},
                {Person.name: "Jane", Person.age: 50},
            ]
        )

    Any columns in the schema that are not present in the data will be
    initialized with ``None`` values.
    """
    if isinstance(data, list):
        col_data = _create_column_wise_data_from_list(schema, data)
    elif isinstance(data, dict):
        col_data = _create_column_wise_data_from_dict(schema, data)
    else:
        raise ValueError("The provided data is not a list or a dict.")

    row_data = zip(*col_data)
    spark_schema = schema.get_structtype()
    dataframe = spark.createDataFrame(row_data, spark_schema)
    return DataSet[schema](dataframe)  # type: ignore


def create_structtype_row(schema: Type[T], data: Dict[Column, Any]) -> Row:
    """Creates a ``Row`` with ``StructType`` schema, where ``data`` is a mapping from
    column to data in the respective column."""
    data_with_string_index = {k.str: v for k, v in data.items()}
    data_converted = {
        k: data_with_string_index[k] if k in data_with_string_index else None
        for k in get_type_hints(schema).keys()
    }
    return Row(**data_converted)


def _create_column_wise_data_from_dict(
    schema: Type[T], data: Dict[Column, List[Any]]
) -> List[List[Any]]:
    """Converts a dict of column to data to a list of lists, where each inner list
    contains the data for a column."""
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

    return col_data


def _create_column_wise_data_from_list(
    schema: Type[T], data: List[Dict[Column, Any]]
) -> List[List[Any]]:
    """Converts a list of dicts of column to data to a list of lists, where each inner
    list contains the data for a column."""
    data_converted = [{k.str: v for k, v in row.items()} for row in data]

    col_data = []
    for col in get_type_hints(schema).keys():
        col_data += [[row[col] if col in row else None for row in data_converted]]

    return col_data
