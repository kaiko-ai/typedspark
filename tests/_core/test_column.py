from dataclasses import dataclass
from threading import Thread
from typing import Annotated

import pandas as pd
import pytest
from pyspark.errors import PySparkRuntimeError
from pyspark.sql import Column as SparkColumn
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import IntegerType, LongType, StringType

from typedspark import Column, ColumnMeta, Schema, StructType
from typedspark._utils.create_dataset import create_partially_filled_dataset


class A(Schema):
    a: Column[LongType]
    b: Column[StringType]


def _make_pyspark_runtime_error(error_class: str) -> PySparkRuntimeError:
    import inspect

    params = inspect.signature(PySparkRuntimeError).parameters
    if "error_class" in params:
        return PySparkRuntimeError(
            error_class=error_class,
            message_parameters={},
        )
    if "errorClass" in params:
        return PySparkRuntimeError(
            errorClass=error_class,
            messageParameters={},
        )

    return PySparkRuntimeError(error_class)


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


@pytest.mark.no_spark_session
def test_column_reference_without_spark_session():
    a = A.a
    assert a.str == "a"


@pytest.mark.no_spark_session
def test_column_active_raises_no_session(monkeypatch: pytest.MonkeyPatch):
    def raise_no_session(cls):
        raise _make_pyspark_runtime_error("NO_ACTIVE_OR_DEFAULT_SESSION")

    monkeypatch.setattr(SparkSession, "active", classmethod(raise_no_session))

    assert repr(A.a) == "Column<'a'> (no active Spark session)"


@pytest.mark.no_spark_session
def test_column_active_reraises_other_errors(monkeypatch: pytest.MonkeyPatch):
    def raise_unexpected(cls):
        raise _make_pyspark_runtime_error("NO_ACTIVE_EXCEPTION")

    monkeypatch.setattr(SparkSession, "active", classmethod(raise_unexpected))

    with pytest.raises(PySparkRuntimeError):
        _ = A.a


def test_column_with_deprecated_dataframe_param(spark: SparkSession):
    df = create_partially_filled_dataset(spark, A, {A.a: [1, 2, 3]})
    Column("a", dataframe=df)


def test_column_comparison_in_thread_without_active_session(spark: SparkSession):
    result: dict[str, SparkColumn] = {}
    error: dict[str, BaseException] = {}

    def compare():
        try:
            result["value"] = A.a == A.b
        except BaseException as exc:  # pragma: no cover
            error["value"] = exc

    thread = Thread(target=compare)
    thread.start()
    thread.join()

    assert "value" not in error
    assert isinstance(result["value"], SparkColumn)


@dataclass
class MyColumnMeta(ColumnMeta):
    primary_key: bool = False


class Persons(Schema):
    id: Annotated[
        Column[LongType],
        MyColumnMeta(
            comment="Identifies the person",
            primary_key=True,
        ),
    ]
    name: Column[StringType]
    age: Column[LongType]


def test_get_metadata():
    assert Persons.get_metadata()["id"] == {
        "comment": "Identifies the person",
        "primary_key": True,
    }


@pytest.mark.no_spark_session
def test_column_repr_no_spark_session():
    spark = SparkSession.getActiveSession()
    if spark is None:
        assert repr(A.a) == "Column<'a'> (no active Spark session)"
    else:
        assert repr(A.a) == "Column<'a'>"


class Cause(Schema):
    source: Column[StringType]


class Values(Schema):
    name: Column[StringType]
    severity: Column[IntegerType]
    cause: Column[StructType[Cause]]


class Actions(Schema):
    consequences: Column[StructType[Values]]


def test_full_path_1():
    assert Actions.consequences.full_path == "consequences"


def test_full_path_2():
    assert Actions.consequences.dtype.schema.severity.full_path == "consequences.severity"


def test_full_path_3():
    assert (
        Actions.consequences.dtype.schema.cause.dtype.schema.source.full_path
        == "consequences.cause.source"
    )
