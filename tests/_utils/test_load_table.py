from typing import Literal

import pytest
from chispa.dataframe_comparer import assert_df_equality  # type: ignore
from pyspark.sql import SparkSession
from pyspark.sql.functions import first
from pyspark.sql.types import IntegerType, StringType

from typedspark import (
    ArrayType,
    Column,
    Databases,
    DecimalType,
    MapType,
    Schema,
    StructType,
    create_empty_dataset,
    load_table,
)
from typedspark._core.datatypes import DayTimeIntervalType
from typedspark._core.literaltype import IntervalType
from typedspark._utils.create_dataset import create_partially_filled_dataset
from typedspark._utils.databases import Catalogs, _get_spark_session
from typedspark._utils.load_table import create_schema


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


def test_databases_with_temp_view(spark):
    df = create_empty_dataset(spark, A)
    df.createOrReplaceTempView("table_a")

    db = Databases(spark)
    for df_loaded, schema in [db.default.table_a(), db.default.table_a.load()]:  # type: ignore
        assert_df_equality(df, df_loaded)
        assert schema.get_structtype() == A.get_structtype()
        assert schema.get_schema_name() == "TableA"
        assert db.default.table_a.str == "table_a"  # type: ignore
        assert db.default.str == "default"  # type: ignore


def _drop_table(spark: SparkSession, table_name: str) -> None:
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")


def test_databases_with_table(spark: SparkSession):
    df = create_empty_dataset(spark, A)
    df.write.saveAsTable("default.table_b")

    try:
        db = Databases(spark)
        df_loaded, schema = db.default.table_b()  # type: ignore

        assert_df_equality(df, df_loaded)
        assert schema.get_structtype() == A.get_structtype()
        assert schema.get_schema_name() == "TableB"
        assert db.default.table_b.str == "default.table_b"  # type: ignore
        assert db.default.str == "default"  # type: ignore
    except Exception as exception:
        _drop_table(spark, "default.table_b")
        raise exception

    _drop_table(spark, "default.table_b")


def test_databases_with_table_name_starting_with_underscore(spark: SparkSession):
    df = create_empty_dataset(spark, A)
    df.write.saveAsTable("default._table_b")

    try:
        db = Databases(spark)
        df_loaded, _ = db.default.u_table_b()  # type: ignore
        assert_df_equality(df, df_loaded)
        assert db.default.u_table_b.str == "default._table_b"  # type: ignore
    except Exception as exception:
        _drop_table(spark, "default._table_b")
        raise exception

    _drop_table(spark, "default._table_b")


def test_databases_with_table_name_starting_with_underscore_with_naming_conflict(
    spark: SparkSession,
):
    df_a = create_empty_dataset(spark, A)
    df_b = create_empty_dataset(spark, B)
    df_a.write.saveAsTable("default._table_b")
    df_b.write.saveAsTable("default.u_table_b")

    try:
        db = Databases(spark)
        df_loaded, _ = db.default.u__table_b()  # type: ignore
        assert_df_equality(df_a, df_loaded)
        assert db.default.u__table_b.str == "default._table_b"  # type: ignore

        df_loaded, _ = db.default.u_table_b()  # type: ignore
        assert_df_equality(df_b, df_loaded)
        assert db.default.u_table_b.str == "default.u_table_b"  # type: ignore
    except Exception as exception:
        _drop_table(spark, "default._table_b")
        _drop_table(spark, "default.u_table_b")
        raise exception

    _drop_table(spark, "default._table_b")
    _drop_table(spark, "default.u_table_b")


def test_catalogs(spark: SparkSession):
    df = create_empty_dataset(spark, A)
    df.write.saveAsTable("spark_catalog.default.table_b")

    try:
        db = Catalogs(spark)
        df_loaded, schema = db.spark_catalog.default.table_b()  # type: ignore

        assert_df_equality(df, df_loaded)
        assert schema.get_structtype() == A.get_structtype()
        assert schema.get_schema_name() == "TableB"
        assert db.spark_catalog.default.table_b.str == "spark_catalog.default.table_b"  # type: ignore  # noqa: E501
    except Exception as exception:
        _drop_table(spark, "spark_catalog.default.table_b")
        raise exception

    _drop_table(spark, "spark_catalog.default.table_b")


def test_get_spark_session(spark: SparkSession):
    res = _get_spark_session(None)

    assert res == spark


@pytest.mark.no_spark_session
def test_get_spark_session_without_spark_session():
    if SparkSession.getActiveSession() is None:
        with pytest.raises(ValueError):
            _get_spark_session(None)
