from chispa.dataframe_comparer import assert_df_equality  # type: ignore
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

from typedspark import (
    ArrayType,
    Column,
    MapType,
    Schema,
    StructType,
    create_empty_dataset,
    load_table,
)


class SubSchema(Schema):
    a: Column[IntegerType]


class A(Schema):
    a: Column[IntegerType]
    b: Column[ArrayType[IntegerType]]
    c: Column[ArrayType[MapType[IntegerType, IntegerType]]]
    d: Column[StructType[SubSchema]]


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
