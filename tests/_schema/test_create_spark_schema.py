from pyspark.sql.types import LongType, StringType, StructField, StructType

from typedspark import Column, Schema


class A(Schema):
    a: Column[LongType]
    b: Column[StringType]


def test_create_spark_schema():
    result = A.get_structtype()
    expected = StructType(
        [
            StructField("a", LongType(), True),
            StructField("b", StringType(), True),
        ]
    )

    assert result == expected
