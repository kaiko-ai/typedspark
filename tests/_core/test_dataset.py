import functools

import pandas as pd
import pytest
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import LongType, StringType

from typedspark import Column, DataSet, Schema
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
    df_class_module = df.__class__.__module__
    assert not df_class_module.startswith("pyspark.sql.connect")
    
    df.distinct()
    cached1: DataSet[A] = df.cache()
    cached2: DataSet[A] = df.persist(StorageLevel.MEMORY_AND_DISK)
    assert isinstance(df.filter(A.a == 1), DataSet)
    assert isinstance(df.where(A.a == 1), DataSet)
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
        DataSetImplements()  # type: ignore


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
