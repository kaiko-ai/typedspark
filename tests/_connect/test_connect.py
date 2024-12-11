import pytest
from chispa.dataframe_comparer import assert_df_equality  # type: ignore
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

from typedspark import (
    Column,
    Schema,
    create_empty_dataset,
    transform_to_schema,
)


class Person(Schema):
    a: Column[IntegerType]
    b: Column[IntegerType]
    c: Column[IntegerType]


class PersonLessData(Schema):
    a: Column[IntegerType]
    b: Column[IntegerType]


def test_spark_connect_session_works(sparkConnect: SparkSession):
    df = sparkConnect.createDataFrame([(14, "Tom"), (23, "Alice"), (16, "Bob")], ["age", "name"])
    assert df.count() == 3

    # notice it isn't regular dataframe but ide doesn't know that
    assert type(df).__module__ == "pyspark.sql.connect.dataframe"


def test_typedspark_does_not_work(sparkConnect: SparkSession):
    df = sparkConnect.createDataFrame([(14, "Tom"), (23, "Alice"), (16, "Bob")], ["age", "name"])
