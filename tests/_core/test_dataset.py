import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, StringType

from typedspark import Column, DataSet, Schema
from typedspark._utils.create_dataset import create_empty_dataset


class A(Schema):
    a: Column[LongType]
    b: Column[StringType]


def create_dataframe(spark: SparkSession, d):
    return spark.createDataFrame(pd.DataFrame(d))


def test_dataset(spark: SparkSession):
    d = dict(
        a=[1, 2, 3],
        b=["a", "b", "c"],
    )
    df = create_dataframe(spark, d)
    DataSet[A](df)


def test_dataset_allow_underscored_columns_not_in_schema(spark: SparkSession):
    d = {"a": [1, 2, 3], "b": ["a", "b", "c"], "__c": [1, 2, 3]}
    df = create_dataframe(spark, d)
    DataSet[A](df)


def test_dataset_single_underscored_column_should_raise(spark: SparkSession):
    d = {"a": [1, 2, 3], "b": ["a", "b", "c"], "_c": [1, 2, 3]}
    df = create_dataframe(spark, d)
    with pytest.raises(TypeError):
        DataSet[A](df)


def test_dataset_missing_colnames(spark: SparkSession):
    d = dict(
        a=[1, 2, 3],
    )
    df = create_dataframe(spark, d)
    with pytest.raises(TypeError):
        DataSet[A](df)


def test_dataset_too_many_colnames(spark: SparkSession):
    d = dict(
        a=[1, 2, 3],
        b=["a", "b", "c"],
        c=[1, 2, 3],
    )
    df = create_dataframe(spark, d)
    with pytest.raises(TypeError):
        DataSet[A](df)


def test_wrong_type(spark: SparkSession):
    d = dict(
        a=[1, 2, 3],
        b=[1, 2, 3],
    )
    df = create_dataframe(spark, d)
    with pytest.raises(TypeError):
        DataSet[A](df)


def test_inherrited_functions(spark: SparkSession):
    df = create_empty_dataset(spark, A)

    df.distinct()
    df.filter(A.a == 1)
    df.orderBy(A.a)
    df.transform(lambda df: df)


def test_inherrited_functions_with_other_dataset(spark: SparkSession):
    df_a = create_empty_dataset(spark, A)
    df_b = create_empty_dataset(spark, A)

    df_a.intersect(df_b)
    df_a.join(df_b, A.a.str)
    df_a.union(df_b)
    df_a.unionByName(df_b)
