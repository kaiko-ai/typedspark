import functools
from typing import Annotated

import pandas as pd
import pytest
from chispa import assert_df_equality  # type: ignore
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import LongType, StringType

from typedspark import Column, DataSet, Schema
from typedspark._core.column_meta import ColumnMeta
from typedspark._core.dataset import DataSetImplements
from typedspark._utils.create_dataset import create_empty_dataset


class A(Schema):
    a: Column[LongType]
    b: Column[StringType]


class B(Schema):
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
    cached1: DataSet[A] = df.cache()
    cached2: DataSet[A] = df.persist(StorageLevel.MEMORY_AND_DISK)
    df.filter(A.a == 1)
    df.orderBy(A.a)
    df.transform(lambda df: df)

    cached1.unpersist(True)
    cached2.unpersist(True)


def test_inherrited_functions_with_other_dataset(spark: SparkSession):
    df_a = create_empty_dataset(spark, A)
    df_b = create_empty_dataset(spark, A)

    df_a.join(df_b, A.a.str)
    df_a.unionByName(df_b)


def test_schema_property_of_dataset(spark: SparkSession):
    df = create_empty_dataset(spark, A)
    assert df.typedspark_schema == A


def test_initialize_dataset_implements(spark: SparkSession):
    with pytest.raises(NotImplementedError):
        DataSetImplements()


def test_reduce(spark: SparkSession):
    functools.reduce(
        DataSet.unionByName,
        [create_empty_dataset(spark, A), create_empty_dataset(spark, A)],
    )


def test_resetting_of_schema_annotations(spark: SparkSession):
    df = create_empty_dataset(spark, A)

    a: DataFrame

    # if no schema is specified, the annotation should be None
    a = DataSet(df)
    assert a._schema_annotations is None

    # when we specify a schema, the class variable will be set to A, but afterwards it should be
    # reset to None again when we initialize a new object without specifying a schema
    DataSet[A]
    a = DataSet(df)
    assert a._schema_annotations is None

    # and then to B
    a = DataSet[B](df)
    assert a._schema_annotations == B

    # and then to None again
    a = DataSet(df)
    assert a._schema_annotations is None


def test_from_dataframe(spark: SparkSession):
    df = spark.createDataFrame([(1, "a"), (2, "b")], ["a", "b"])
    ds = DataSet[A].from_dataframe(df)

    assert isinstance(ds, DataSet)
    assert_df_equality(ds, df)

    df_2 = ds.to_dataframe()

    assert isinstance(df_2, DataFrame)
    assert_df_equality(df_2, df)


class Person(Schema):
    name: Annotated[Column[StringType], ColumnMeta(external_name="first-name")]
    age: Column[LongType]


def test_from_dataframe_with_external_name(spark: SparkSession):
    df = spark.createDataFrame([("Alice", 1), ("Bob", 2)], ["first-name", "age"])
    ds = DataSet[Person].from_dataframe(df)

    assert isinstance(ds, DataSet)
    assert ds.columns == ["name", "age"]

    df_2 = ds.to_dataframe()
    assert isinstance(df_2, DataFrame)
    assert df_2.columns == ["first-name", "age"]
    assert_df_equality(df_2, df)
