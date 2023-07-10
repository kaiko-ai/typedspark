from typing import Literal

import pytest
from chispa.dataframe_comparer import assert_df_equality  # type: ignore
from pyspark.sql import SparkSession
from pyspark.sql.functions import first
from pyspark.sql.types import IntegerType, StringType

from typedspark import (
    ArrayType,
    Column,
    DecimalType,
    MapType,
    Schema,
    StructType,
    create_empty_dataset,
    load_table,
)
from typedspark._utils.create_dataset import create_partially_filled_dataset
from typedspark._utils.load_table import create_schema
from typedspark._core.datatypes import DayTimeIntervalType
from typedspark._core.literaltype import IntervalType


class SubSchema(Schema):
    a: Column[IntegerType]


class A(Schema):
    a: Column[IntegerType]
    b: Column[ArrayType[IntegerType]]
    c: Column[ArrayType[MapType[IntegerType, IntegerType]]]
    d: Column[DayTimeIntervalType[IntervalType.HOUR, IntervalType.MINUTE]]
    e: Column[DecimalType[Literal[7], Literal[2]]]
    value_container: Column[StructType[SubSchema]]


def test_load_table(spark: SparkSession) -> None:
    df = create_empty_dataset(spark, A)
    df.createOrReplaceTempView("temp")

    df_loaded, schema = load_table(spark, "temp")

    assert_df_equality(df, df_loaded)
    assert schema.get_structtype() == A.get_structtype()
    assert schema.get_schema_name() != "A"


def test_load_table_with_schema_name(spark: SparkSession) -> None:
    df = create_empty_dataset(spark, A)
    df.createOrReplaceTempView("temp")

    df_loaded, schema = load_table(spark, "temp", schema_name="A")

    assert_df_equality(df, df_loaded)
    assert schema.get_structtype() == A.get_structtype()
    assert schema.get_schema_name() == "A"


class B(Schema):
    a: Column[StringType]
    b: Column[IntegerType]
    c: Column[StringType]


def test_create_schema(spark: SparkSession) -> None:
    df = (
        create_partially_filled_dataset(
            spark,
            B,
            {
                B.a: ["a", "b!!", "c", "a", "b!!", "c", "a", "b!!", "c"],
                B.b: [1, 1, 1, 2, 2, 2, 3, 3, 3],
                B.c: ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota"],
            },
        )
        .groupby(B.b)
        .pivot(B.a.str)
        .agg(first(B.c))
    )

    df, MySchema = create_schema(df, "B")

    assert MySchema.get_schema_name() == "B"
    assert "a" in MySchema.all_column_names()
    assert "b__" in MySchema.all_column_names()
    assert "c" in MySchema.all_column_names()


def test_create_schema_with_duplicated_column_names(spark: SparkSession) -> None:
    df = (
        create_partially_filled_dataset(
            spark,
            B,
            {
                B.a: ["a", "b??", "c", "a", "b!!", "c", "a", "b!!", "c"],
                B.b: [1, 1, 1, 2, 2, 2, 3, 3, 3],
                B.c: ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota"],
            },
        )
        .groupby(B.b)
        .pivot(B.a.str)
        .agg(first(B.c))
    )

    with pytest.raises(ValueError):
        create_schema(df, "B")


def test_name_of_structtype_schema(spark):
    df = create_empty_dataset(spark, A)
    df, MySchema = create_schema(df, "A")

    assert MySchema.value_container.dtype.schema.get_schema_name() == "ValueContainer"
