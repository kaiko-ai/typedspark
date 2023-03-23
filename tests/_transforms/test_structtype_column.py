import pytest
from chispa.dataframe_comparer import assert_df_equality  # type: ignore
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

from typedspark import Column, Schema, StructType, structtype_column
from typedspark._transforms.transform_to_schema import transform_to_schema
from typedspark._utils.create_dataset import create_partially_filled_dataset


class SubSchema(Schema):
    a: Column[IntegerType]
    b: Column[IntegerType]


class MainSchema(Schema):
    a: Column[IntegerType]
    b: Column[StructType[SubSchema]]


def test_structtype_column(spark: SparkSession):
    df = create_partially_filled_dataset(spark, MainSchema, {MainSchema.a: [1, 2, 3]})
    observed = transform_to_schema(
        df,
        MainSchema,
        {
            MainSchema.b: structtype_column(
                SubSchema,
                {SubSchema.a: MainSchema.a + 2, SubSchema.b: MainSchema.a + 4},
            )
        },
    )
    expected = create_partially_filled_dataset(
        spark,
        MainSchema,
        {
            MainSchema.a: [1, 2, 3],
            MainSchema.b: create_partially_filled_dataset(
                spark, SubSchema, {SubSchema.a: [3, 4, 5], SubSchema.b: [5, 6, 7]}
            ).collect(),
        },
    )
    assert_df_equality(observed, expected, ignore_nullable=True)


def test_structtype_column_different_column_order(spark: SparkSession):
    df = create_partially_filled_dataset(spark, MainSchema, {MainSchema.a: [1, 2, 3]})
    observed = transform_to_schema(
        df,
        MainSchema,
        {
            MainSchema.b: structtype_column(
                SubSchema,
                {SubSchema.b: MainSchema.a + 4, SubSchema.a: MainSchema.a + 2},
            )
        },
    )
    expected = create_partially_filled_dataset(
        spark,
        MainSchema,
        {
            MainSchema.a: [1, 2, 3],
            MainSchema.b: create_partially_filled_dataset(
                spark, SubSchema, {SubSchema.a: [3, 4, 5], SubSchema.b: [5, 6, 7]}
            ).collect(),
        },
    )
    assert_df_equality(observed, expected, ignore_nullable=True)


def test_structtype_column_partial(spark: SparkSession):
    df = create_partially_filled_dataset(spark, MainSchema, {MainSchema.a: [1, 2, 3]})
    observed = transform_to_schema(
        df,
        MainSchema,
        {
            MainSchema.b: structtype_column(
                SubSchema,
                {SubSchema.a: MainSchema.a + 2},
                fill_unspecified_columns_with_nulls=True,
            )
        },
    )
    expected = create_partially_filled_dataset(
        spark,
        MainSchema,
        {
            MainSchema.a: [1, 2, 3],
            MainSchema.b: create_partially_filled_dataset(
                spark,
                SubSchema,
                {SubSchema.a: [3, 4, 5], SubSchema.b: [None, None, None]},
            ).collect(),
        },
    )
    assert_df_equality(observed, expected, ignore_nullable=True)


def test_structtype_column_with_double_column(spark: SparkSession):
    df = create_partially_filled_dataset(spark, MainSchema, {MainSchema.a: [1, 2, 3]})
    with pytest.raises(ValueError):
        transform_to_schema(
            df,
            MainSchema,
            {
                MainSchema.b: structtype_column(
                    SubSchema,
                    {SubSchema.a: MainSchema.a + 2, SubSchema.a: MainSchema.a + 2},
                )
            },
        )
