import pytest
from chispa.dataframe_comparer import assert_df_equality  # type: ignore
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType

from typedspark import (
    Column,
    Schema,
    StructType,
    create_partially_filled_dataset,
    register_schema_to_dataset,
)
from typedspark._core.spark_imports import SPARK_CONNECT, AnalysisException
from typedspark._utils.register_schema_to_dataset import register_schema_to_dataset_with_alias


class Pets(Schema):
    name: Column[StringType]
    age: Column[IntegerType]


class Person(Schema):
    a: Column[IntegerType]
    b: Column[IntegerType]
    pets: Column[StructType[Pets]]


class Job(Schema):
    a: Column[IntegerType]
    c: Column[IntegerType]


class Age(Schema):
    age_1: Column[IntegerType]
    age_2: Column[IntegerType]


def test_register_schema_to_dataset(spark: SparkSession):
    df_a = create_partially_filled_dataset(spark, Person, {Person.a: [1, 2, 3]})
    df_b = create_partially_filled_dataset(spark, Job, {Job.a: [1, 2, 3]})

    with pytest.raises(AnalysisException):
        df_a.join(df_b, Person.a == Job.a).show()

    person = register_schema_to_dataset(df_a, Person)
    job = register_schema_to_dataset(df_b, Job)

    assert person.get_schema_name() == "Person"
    assert hash(person.a) != hash(Person.a)

    df_a.join(df_b, person.a == job.a)


def test_register_schema_to_dataset_with_alias(spark: SparkSession):
    df = create_partially_filled_dataset(
        spark,
        Person,
        {
            Person.a: [1, 2, 3],
            Person.b: [1, 2, 3],
            Person.pets: create_partially_filled_dataset(
                spark,
                Pets,
                {
                    Pets.name: ["Bobby", "Bobby", "Bobby"],
                    Pets.age: [10, 20, 30],
                },
            ).collect(),
        },
    )

    def self_join_without_register_schema_to_dataset_with_alias():
        df_a = df.alias("a")
        df_b = df.alias("b")
        schema_a = register_schema_to_dataset(df_a, Person)
        schema_b = register_schema_to_dataset(df_b, Person)
        df_a.join(df_b, schema_a.a == schema_b.b).show()

    # there seems to be a discrepancy between spark and spark connect here
    if SPARK_CONNECT:
        self_join_without_register_schema_to_dataset_with_alias()
    else:
        with pytest.raises(AnalysisException):
            self_join_without_register_schema_to_dataset_with_alias()

    # the following is the way it works with regular spark
    df_a, schema_a = register_schema_to_dataset_with_alias(df, Person, "a")
    df_b, schema_b = register_schema_to_dataset_with_alias(df, Person, "b")
    joined = df_a.join(df_b, schema_a.a == schema_b.b)

    res = joined.select(
        schema_a.pets.dtype.schema.age.alias(Age.age_1.str),
        schema_b.pets.dtype.schema.age.alias(Age.age_2.str),
    )

    expected = create_partially_filled_dataset(
        spark, Age, {Age.age_1: [10, 20, 30], Age.age_2: [10, 20, 30]}
    )

    assert_df_equality(res, expected)
