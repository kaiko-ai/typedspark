import functools
from typing import cast

import pandas as pd
import pytest
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, StringType, TimestampType

from typedspark import Column, DataSet, Schema
from typedspark._core.dataset import DataSetImplements
from typedspark._utils.create_dataset import create_empty_dataset


class A(Schema):
    a: Column[LongType]
    b: Column[StringType]


class B(Schema):
    a: Column[LongType]
    b: Column[StringType]


class Rate(Schema):
    timestamp: Column[TimestampType]
    value: Column[LongType]


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

    assert isinstance(df.distinct(), DataSet)
    assert isinstance(df.dropDuplicates(["a"]), DataSet)
    assert isinstance(df.drop_duplicates(["a"]), DataSet)
    assert isinstance(df.dropna(), DataSet)
    assert isinstance(df.fillna(1), DataSet)
    assert isinstance(df.replace(1, 2), DataSet)
    assert isinstance(df.repartition(2), DataSet)
    assert isinstance(df.repartition(2, A.a), DataSet)
    assert isinstance(df.repartition(A.a), DataSet)
    assert isinstance(DataSetImplements.repartition(df, A.a), DataSet)
    assert isinstance(df.repartition(cast(int, None), A.a), DataSet)
    assert isinstance(DataSetImplements.repartition(df, cast(int, None), A.a), DataSet)
    assert isinstance(df.repartitionByRange(2, A.a), DataSet)
    assert isinstance(df.repartitionByRange(A.a), DataSet)
    assert isinstance(DataSetImplements.repartitionByRange(df, A.a), DataSet)
    assert isinstance(df.repartitionByRange(cast(int, None), A.a), DataSet)
    assert isinstance(DataSetImplements.repartitionByRange(df, cast(int, None), A.a), DataSet)
    assert isinstance(df.limit(1), DataSet)
    assert isinstance(df.coalesce(1), DataSet)
    assert isinstance(df.sample(True, 0.5, 1), DataSet)
    assert isinstance(df.sampleBy("a", {1: 0.5}), DataSet)
    assert isinstance(df.sort(A.a), DataSet)
    assert isinstance(df.sortWithinPartitions(A.a), DataSet)
    cached1: DataSet[A] = df.cache()
    cached2: DataSet[A] = df.persist(StorageLevel.MEMORY_AND_DISK)
    assert isinstance(df.filter(A.a == 1), DataSet)
    assert isinstance(df.where(A.a == 1), DataSet)
    assert isinstance(df.orderBy(A.a), DataSet)
    assert isinstance(df.hint("broadcast"), DataSet)
    if hasattr(DataFrame, "observe"):
        assert isinstance(df.observe("obs", F.count("*").alias("cnt")), DataSet)
    assert isinstance(df.transform(lambda df: df), DataSet)

    cached1.unpersist(True)
    cached2.unpersist(True)


def test_inherrited_checkpoint_functions(spark: SparkSession, tmp_path):
    df = create_empty_dataset(spark, A)

    spark.sparkContext.setCheckpointDir(str(tmp_path))
    assert isinstance(df.localCheckpoint(), DataSet)
    assert isinstance(df.checkpoint(), DataSet)


def test_local_checkpoint_storage_level_fallback(spark: SparkSession, tmp_path, monkeypatch):
    df = create_empty_dataset(spark, A)
    spark.sparkContext.setCheckpointDir(str(tmp_path))
    base_df_cls = next(cls for cls in type(df).mro() if cls.__name__ == "DataFrame")
    original = getattr(base_df_cls, "localCheckpoint")
    flags = {"raised": False, "called_fallback": False}

    def legacy_local_checkpoint(self, eager=True, storageLevel=None):
        if storageLevel is not None:
            flags["raised"] = True
            raise TypeError("legacy localCheckpoint without storageLevel support")
        flags["called_fallback"] = True
        return original(self, eager)

    monkeypatch.setattr(base_df_cls, "localCheckpoint", legacy_local_checkpoint)

    assert isinstance(df.localCheckpoint(storageLevel=StorageLevel.MEMORY_AND_DISK), DataSet)
    assert isinstance(
        DataSetImplements.localCheckpoint(df, storageLevel=StorageLevel.MEMORY_AND_DISK),
        DataSet,
    )
    assert flags["raised"]
    assert flags["called_fallback"]

    original_ds_local_checkpoint = DataSetImplements.localCheckpoint

    def legacy_ds_local_checkpoint(self, eager=True, storageLevel=None):
        if storageLevel is not None:
            raise TypeError("legacy DataSetImplements localCheckpoint without storageLevel support")
        return original_ds_local_checkpoint(self, eager)

    monkeypatch.setattr(DataSetImplements, "localCheckpoint", legacy_ds_local_checkpoint)
    assert isinstance(df.localCheckpoint(storageLevel=StorageLevel.MEMORY_AND_DISK), DataSet)


def test_inherrited_drop_duplicates_within_watermark(spark: SparkSession):
    if not hasattr(DataFrame, "dropDuplicatesWithinWatermark"):
        pytest.skip("dropDuplicatesWithinWatermark not available in this Spark version")

    stream_df = spark.readStream.format("rate").load()
    watermarked = stream_df.withWatermark("timestamp", "1 minute")
    ds = DataSet[Rate](watermarked)
    assert isinstance(ds.dropDuplicatesWithinWatermark(["value"]), DataSet)


def test_inherrited_functions_with_other_dataset(spark: SparkSession):
    df_a = create_empty_dataset(spark, A)
    df_b = create_empty_dataset(spark, A)
    df_plain = df_b.select("a", "b")

    df_a.join(df_b, A.a.str)
    assert isinstance(df_a.union(df_b), DataSet)
    assert isinstance(df_a.unionAll(df_b), DataSet)
    assert isinstance(df_a.union(df_plain), DataFrame)
    assert isinstance(df_a.unionAll(df_plain), DataFrame)
    df_a.unionByName(df_b)
    assert isinstance(df_a.intersect(df_b), DataSet)
    assert isinstance(df_a.intersectAll(df_b), DataSet)
    assert isinstance(df_a.exceptAll(df_b), DataSet)
    assert isinstance(df_a.subtract(df_b), DataSet)
    assert isinstance(df_a.intersect(df_plain), DataFrame)
    assert isinstance(df_a.intersectAll(df_plain), DataFrame)
    assert isinstance(df_a.exceptAll(df_plain), DataFrame)
    assert isinstance(df_a.subtract(df_plain), DataFrame)

    assert isinstance(DataSetImplements.union(df_a, df_plain), DataFrame)
    assert isinstance(DataSetImplements.unionAll(df_a, df_plain), DataFrame)
    assert isinstance(DataSetImplements.intersect(df_a, df_plain), DataFrame)
    assert isinstance(DataSetImplements.intersectAll(df_a, df_plain), DataFrame)
    assert isinstance(DataSetImplements.exceptAll(df_a, df_plain), DataFrame)
    assert isinstance(DataSetImplements.subtract(df_a, df_plain), DataFrame)


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
