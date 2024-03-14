from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

from typedspark._core.column import Column
from typedspark._schema.schema import Schema
from typedspark._utils.create_dataset import create_empty_dataset
from typedspark._utils.to_dataset import to_dataset


class A(Schema):
    a: Column[IntegerType]


def test_to_dataset(spark: SparkSession):
    df = create_empty_dataset(spark, A)
    df = to_dataset(df, A)
