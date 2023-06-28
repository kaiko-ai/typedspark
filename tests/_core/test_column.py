import pandas as pd
import pytest
from chispa import assert_df_equality
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import LongType, StringType

from typedspark import Column, Schema
from typedspark._utils.create_dataset import create_partially_filled_dataset


class A(Schema):
    a: Column[LongType]
    b: Column[StringType]


def test_column(spark: SparkSession):
    (
        spark.createDataFrame(
            pd.DataFrame(
                dict(
                    a=[1, 2, 3],
                )
            )
        )
        .filter(A.a == 1)
        .withColumn(A.b.str, lit("a"))
    )


def test_column_doesnt_exist():
    with pytest.raises(TypeError):
        A.z


def test_column_reference_without_spark_session():
    a = A.a
    assert a.str == "a"


def test_column_with_deprecated_dataframe_param(spark: SparkSession):
    df = create_partially_filled_dataset(spark, A, {A.a: [1, 2, 3]})
    Column("a", dataframe=df)
