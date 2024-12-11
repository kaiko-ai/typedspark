from pyspark.sql import SparkSession
from pyspark.sql.types import LongType

from typedspark import Column, Schema


class A(Schema):
    a: Column[LongType]
    b: Column[LongType]
    c: Column[LongType]


def test_regular_spark_works(spark: SparkSession):
    df = spark.createDataFrame([(14, "Tom"), (23, "Alice"), (16, "Bob")], ["age", "name"])
    assert df.count() == 3
    assert type(df).__module__ == "pyspark.sql.dataframe"


def test_spark_connect_works(sparkConnect: SparkSession):
    df = sparkConnect.createDataFrame([(14, "Tom"), (23, "Alice"), (16, "Bob")], ["age", "name"])
    assert df.count() == 3
    assert type(df).__module__ == "pyspark.sql.connect.dataframe"
