from dataclasses import dataclass
from typing import Annotated

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import IntegerType, LongType, StringType

from typedspark import Column, ColumnMeta, Schema, StructType
from typedspark._utils.create_dataset import create_partially_filled_dataset


class A(Schema):
    a: Column[LongType]
    b: Column[StringType]


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


def test_column_with_deprecated_dataframe_param(spark: SparkSession):
    df = create_partially_filled_dataset(spark, A, {A.a: [1, 2, 3]})
    Column("a", dataframe=df)


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


class Values(Schema):
    name: Column[StringType]
    severity: Column[IntegerType]


class Actions(Schema):
    consequeces: Column[StructType[Values]]


def test_full_path():
    assert Actions.consequences.dtype.schema.severity.full_path == "consequeces.severity"
