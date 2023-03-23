from typing import Annotated

from pyspark.sql import SparkSession
from pyspark.sql.types import LongType

from typedspark import Column, DataSet, Schema, create_empty_dataset
from typedspark._core.column_meta import ColumnMeta


class A(Schema):
    a: Annotated[Column[LongType], ColumnMeta(comment="test")]
    b: Column[LongType]


def test_add_schema_metadata(spark: SparkSession):
    df: DataSet[A] = create_empty_dataset(spark, A, 1)
    assert df.schema["a"].metadata == {"comment": "test"}
    assert df.schema["b"].metadata == {}


class B(Schema):
    a: Column[LongType]
    b: Annotated[Column[LongType], ColumnMeta(comment="test")]


def test_refresh_metadata(spark: SparkSession):
    df_a = create_empty_dataset(spark, A, 1)
    df_b = DataSet[B](df_a)
    assert df_b.schema["a"].metadata == {}
    assert df_b.schema["b"].metadata == {"comment": "test"}
