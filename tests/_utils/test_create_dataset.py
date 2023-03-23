from decimal import Decimal
from typing import Literal

import pytest
from chispa.dataframe_comparer import assert_df_equality  # type: ignore
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType

from typedspark import (
    ArrayType,
    Column,
    DataSet,
    MapType,
    Schema,
    StructType,
    create_empty_dataset,
    create_partially_filled_dataset,
)
from typedspark._core.datatypes import DecimalType


class A(Schema):
    a: Column[DecimalType[Literal[38], Literal[18]]]
    b: Column[StringType]


def test_create_empty_dataset(spark: SparkSession):
    n_rows = 2
    result: DataSet[A] = create_empty_dataset(spark, A, n_rows)

    spark_schema = A.get_structtype()
    data = [(None, None), (None, None)]
    expected = spark.createDataFrame(data, spark_schema)

    assert_df_equality(result, expected)


def test_create_partially_filled_dataset(spark: SparkSession):
    data = {A.a: [Decimal(x) for x in [1, 2, 3]]}
    result: DataSet[A] = create_partially_filled_dataset(spark, A, data)

    spark_schema = A.get_structtype()
    row_data = [(Decimal(1), None), (Decimal(2), None), (Decimal(3), None)]
    expected = spark.createDataFrame(row_data, spark_schema)

    assert_df_equality(result, expected)


def test_create_partially_filled_dataset_with_different_number_of_rows(
    spark: SparkSession,
):
    with pytest.raises(ValueError):
        create_partially_filled_dataset(spark, A, {A.a: [1], A.b: ["a", "b"]})


class B(Schema):
    a: Column[ArrayType[StringType]]
    b: Column[MapType[StringType, StringType]]
    c: Column[StructType[A]]


def test_create_empty_dataset_with_complex_data(spark: SparkSession):
    df_a = create_partially_filled_dataset(spark, A, {A.a: [Decimal(x) for x in [1, 2, 3]]})

    create_partially_filled_dataset(
        spark,
        B,
        {
            B.a: [["a"], ["b", "c"], ["d"]],
            B.b: [{"a": "1"}, {"b": "2", "c": "3"}, {"d": "4"}],
            B.c: df_a.collect(),
        },
    )
