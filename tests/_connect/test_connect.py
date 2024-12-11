from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
import pytest

from typedspark import Column, Schema, transform_to_schema


class A(Schema):
    a: Column[IntegerType]
    b: Column[IntegerType]
    c: Column[IntegerType]


class B(Schema):
    a: Column[IntegerType]
    b: Column[IntegerType]


def test_regular_spark_works(spark: SparkSession):
    df = spark.createDataFrame([(14, "Tom"), (23, "Alice"), (16, "Bob")], ["age", "name"])
    assert df.count() == 3
    assert type(df).__module__ == "pyspark.sql.dataframe"


def test_spark_connect_works(sparkConnect: SparkSession):
    df = sparkConnect.createDataFrame([(14, "Tom"), (23, "Alice"), (16, "Bob")], ["age", "name"])
    assert df.count() == 3
    assert type(df).__module__ == "pyspark.sql.connect.dataframe"


@pytest.mark.skip(reason="in development")
def test_transform_to_schema_works(sparkConnect: SparkSession):
    # this test is getting stuck currently,
    # cuz of mismatch DataFrame types spark.sql.DataFrame and spark.connect.dataframe.DataFrame
    df = sparkConnect.createDataFrame([(14, 23, 16)], ["a", "b", "c"])
    typed_df = transform_to_schema(df, A)
    assert typed_df.count() == 1
