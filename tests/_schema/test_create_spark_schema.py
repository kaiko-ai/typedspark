import pytest
from pyspark.sql.types import LongType, StringType, StructField, StructType

from typedspark import Column, Schema


class A(Schema):
    a: Column[LongType]
    b: Column[StringType]


class B(Schema):
    a: Column


def test_create_spark_schema():
    result = A.get_structtype()
    expected = StructType(
        [
            StructField("a", LongType(), True),
            StructField("b", StringType(), True),
        ]
    )

    assert result == expected


def test_create_spark_schema_with_faulty_schema():
    with pytest.raises(TypeError):
        B.get_structtype()
