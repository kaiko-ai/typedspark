import pytest
from pyspark.sql.types import LongType

from typedspark import Column, Schema, create_empty_dataset


def _get_classic_classes():
    try:
        from pyspark.sql.classic.column import Column as ClassicColumn
        from pyspark.sql.classic.dataframe import DataFrame as ClassicDataFrame
    except Exception:  # pragma: no cover - pyspark < 4
        return None

    return ClassicDataFrame, ClassicColumn


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
