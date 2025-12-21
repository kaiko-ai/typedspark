import os

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType

from typedspark import Column, Schema, create_empty_dataset


def _get_classic_classes():
    try:
        from pyspark.sql.classic.column import Column as ClassicColumn
        from pyspark.sql.classic.dataframe import DataFrame as ClassicDataFrame
    except Exception:  # pragma: no cover - pyspark < 4
        return None

    return ClassicDataFrame, ClassicColumn


def _get_connect_classes():
    try:
        from pyspark.sql.connect.column import Column as ConnectColumn
        from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame
    except Exception:  # pragma: no cover - pyspark < 4 or missing deps
        return None

    return ConnectDataFrame, ConnectColumn


class A(Schema):
    a: Column[LongType]


def test_dataset_preserves_classic_dataframe(spark):
    classic = _get_classic_classes()
    if classic is None:
        pytest.skip("PySpark classic classes not available")
    classic_df, _classic_col = classic

    df = create_empty_dataset(spark, A, 1)
    assert isinstance(df, classic_df)


def test_column_preserves_classic_column(spark):
    classic = _get_classic_classes()
    if classic is None:
        pytest.skip("PySpark classic classes not available")
    _classic_df, classic_col = classic

    _ = spark  # ensure active session for column creation
    assert isinstance(A.a, classic_col)


def test_dataset_preserves_connect_dataframe():
    connect = _get_connect_classes()
    if connect is None:
        pytest.skip("PySpark connect classes not available")

    connect_url = os.environ.get("SPARK_CONNECT_URL")
    if not connect_url:
        pytest.skip("SPARK_CONNECT_URL not set; skipping Spark Connect test")

    connect_df, connect_col = connect
    spark = SparkSession.builder.remote(connect_url).getOrCreate()
    try:
        df = create_empty_dataset(spark, A, 1)
        assert isinstance(df, connect_df)
        assert isinstance(A.a, connect_col)
    finally:
        spark.stop()
