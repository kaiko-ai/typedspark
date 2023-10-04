import pytest
from chispa.dataframe_comparer import assert_df_equality  # type: ignore
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType

from typedspark import (
    Column,
    Schema,
    create_empty_dataset,
    register_schema_to_dataset,
    transform_to_schema,
)
from typedspark._utils.create_dataset import create_partially_filled_dataset


class Person(Schema):
    a: Column[IntegerType]
    b: Column[IntegerType]
    c: Column[IntegerType]


class PersonLessData(Schema):
    a: Column[IntegerType]
    b: Column[IntegerType]


class PersonDifferentData(Schema):
    a: Column[IntegerType]
    d: Column[IntegerType]


def test_transform_to_schema_without_transformations(spark: SparkSession):
    df = create_empty_dataset(spark, Person)
    observed = transform_to_schema(df, PersonLessData)
    expected = create_empty_dataset(spark, PersonLessData)
    assert_df_equality(observed, expected)


def test_transform_to_schema_with_transformation(spark: SparkSession):
    df = create_partially_filled_dataset(spark, Person, {Person.c: [1, 2, 3]})
    observed = transform_to_schema(df, PersonDifferentData, {PersonDifferentData.d: Person.c + 3})
    expected = create_partially_filled_dataset(
        spark, PersonDifferentData, {PersonDifferentData.d: [4, 5, 6]}
    )
    assert_df_equality(observed, expected)


def test_transform_to_schema_with_missing_column(spark):
    df = create_partially_filled_dataset(spark, Person, {Person.c: [1, 2, 3]}).drop(Person.a)
    with pytest.raises(Exception):
        transform_to_schema(df, PersonDifferentData, {PersonDifferentData.d: Person.c + 3})

    observed = transform_to_schema(
        df,
        PersonDifferentData,
        {PersonDifferentData.d: Person.c + 3},
        fill_unspecified_columns_with_nulls=True,
    )

    expected = create_partially_filled_dataset(
        spark,
        PersonDifferentData,
        {PersonDifferentData.d: [4, 5, 6]},
    )
    assert_df_equality(observed, expected)


def test_transform_to_schema_with_pre_existing_column(spark):
    df = create_partially_filled_dataset(spark, Person, {Person.a: [0, 1, 2], Person.c: [1, 2, 3]})

    observed = transform_to_schema(
        df,
        PersonDifferentData,
        {PersonDifferentData.d: Person.c + 3},
        fill_unspecified_columns_with_nulls=True,
    )

    expected = create_partially_filled_dataset(
        spark,
        PersonDifferentData,
        {PersonDifferentData.a: [0, 1, 2], PersonDifferentData.d: [4, 5, 6]},
    )
    assert_df_equality(observed, expected)


class PersonA(Schema):
    name: Column[StringType]
    age: Column[StringType]


class PersonB(Schema):
    name: Column[StringType]
    age: Column[StringType]


def test_transform_to_schema_with_column_disambiguation(spark: SparkSession):
    df_a = create_partially_filled_dataset(
        spark,
        PersonA,
        {PersonA.name: ["John", "Jane", "Bob"], PersonA.age: [30, 40, 50]},
    )
    df_b = create_partially_filled_dataset(
        spark,
        PersonB,
        {PersonB.name: ["John", "Jane", "Bob"], PersonB.age: [31, 41, 51]},
    )

    person_a = register_schema_to_dataset(df_a, PersonA)
    person_b = register_schema_to_dataset(df_b, PersonB)

    with pytest.raises(ValueError):
        transform_to_schema(
            df_a.join(df_b, person_a.name == person_b.name),
            PersonA,
            {
                person_a.age: person_a.age + 3,
                person_b.age: person_b.age + 5,
            },
        )

    with pytest.raises(ValueError):
        transform_to_schema(
            df_a.join(df_b, person_a.name == person_b.name),
            PersonA,
            {
                PersonA.age: person_a.age,
            },
        )

    res = transform_to_schema(
        df_a.join(df_b, person_a.name == person_b.name),
        PersonA,
        {
            PersonA.name: person_a.name,
            PersonB.age: person_b.age,
        },
    )
    expected = create_partially_filled_dataset(
        spark,
        PersonA,
        {PersonA.name: ["John", "Jane", "Bob"], PersonA.age: [31, 41, 51]},
    )
    assert_df_equality(res, expected, ignore_row_order=True)


def test_transform_to_schema_with_double_column(spark: SparkSession):
    df = create_partially_filled_dataset(spark, Person, {Person.a: [1, 2, 3], Person.b: [1, 2, 3]})

    with pytest.raises(ValueError):
        transform_to_schema(
            df,
            Person,
            {
                Person.a: Person.a + 3,
                Person.a: Person.a + 5,
            },
        )


def test_transform_to_schema_sequential(spark: SparkSession):
    df = create_partially_filled_dataset(spark, Person, {Person.a: [1, 2, 3], Person.b: [1, 2, 3]})

    observed = transform_to_schema(
        df,
        Person,
        {
            Person.a: Person.a + 3,
            Person.b: Person.a + 5,
        },
    )

    expected = create_partially_filled_dataset(
        spark, Person, {Person.a: [4, 5, 6], Person.b: [9, 10, 11]}
    )

    assert_df_equality(observed, expected)
