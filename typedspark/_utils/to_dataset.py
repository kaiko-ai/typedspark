from typing import Tuple, Type, TypeVar

from pyspark.sql import DataFrame

from typedspark._core.dataset import DataSet
from typedspark._schema.schema import Schema
from typedspark._transforms.transform_to_schema import transform_to_schema
from typedspark._utils.register_schema_to_dataset import register_schema_to_dataset
from typedspark._utils.replace_illegal_column_names import replace_illegal_column_names

T = TypeVar("T", bound=Schema)


def to_dataset(df: DataFrame, schema: Type[T]) -> Tuple[DataSet[T], Type[T]]:
    """Converts a DataFrame to a DataSet and registers the Schema to the DataSet.

    Also replaces "illegal" characters in the DataFrame's colnames (.e.g "test-result"
    -> "test_result"), so they're compatible with the Schema (after all, Python doesn't allow for
    characters such as dashes in attribute names).
    """
    df = replace_illegal_column_names(df)
    ds = transform_to_schema(df, schema)
    schema = register_schema_to_dataset(ds, schema)
    return ds, schema
