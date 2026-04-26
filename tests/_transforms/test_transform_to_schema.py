import pytest
from chispa.dataframe_comparer import assert_df_equality  # type: ignore
from pyspark.errors import AnalysisException
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType

from typedspark import (
    ArrayType,
    Column,
    MapType,
    Schema,
    StructType,
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


def test_transform_to_schema_parallel(spark: SparkSession):
    df = create_partially_filled_dataset(spark, Person, {Person.a: [1, 2, 3], Person.b: [1, 2, 3]})

    observed = transform_to_schema(
        df,
        Person,
        {
            Person.a: Person.a + 3,
            Person.b: Person.a + 5,
        },
        run_sequentially=False,
    )

    expected = create_partially_filled_dataset(
        spark, Person, {Person.a: [4, 5, 6], Person.b: [6, 7, 8]}
    )

    assert_df_equality(observed, expected)


# Schemas shared across multiple fill_unspecified_inner_fields tests


class InnerFull(Schema):
    x: Column[IntegerType]
    y: Column[IntegerType]


class InnerPartial(Schema):
    x: Column[IntegerType]


class TopStructPartial(Schema):
    id: Column[IntegerType]
    inner: Column[StructType[InnerPartial]]


def test_fill_unspecified_inner_fields_simple_missing_field(spark: SparkSession):
    class TopStructFull(Schema):
        id: Column[IntegerType]
        inner: Column[StructType[InnerFull]]

    ds = create_partially_filled_dataset(
        spark,
        TopStructPartial,
        {TopStructPartial.id: [1, 2], TopStructPartial.inner: [{"x": 10}, {"x": 20}]},
    )
    observed = transform_to_schema(
        ds, TopStructFull, {}, fill_unspecified_inner_fields_with_nulls=True
    )
    expected = create_partially_filled_dataset(
        spark,
        TopStructFull,
        {
            TopStructFull.id: [1, 2],
            TopStructFull.inner: [{"x": 10, "y": None}, {"x": 20, "y": None}],
        },
    )
    assert_df_equality(observed, expected)


def test_fill_unspecified_inner_fields_deeply_nested(spark: SparkSession):
    class WrapperFull(Schema):
        n: Column[IntegerType]
        leaf: Column[StructType[InnerFull]]

    class WrapperPartial(Schema):
        n: Column[IntegerType]
        leaf: Column[StructType[InnerPartial]]

    class TopStructDeepFull(Schema):
        id: Column[IntegerType]
        wrapper: Column[StructType[WrapperFull]]

    class TopStructDeepPartial(Schema):
        id: Column[IntegerType]
        wrapper: Column[StructType[WrapperPartial]]

    ds = create_partially_filled_dataset(
        spark,
        TopStructDeepPartial,
        {
            TopStructDeepPartial.id: [1, 2],
            TopStructDeepPartial.wrapper: [
                {"n": 1, "leaf": {"x": 10}},
                {"n": 2, "leaf": {"x": 20}},
            ],
        },
    )
    observed = transform_to_schema(
        ds, TopStructDeepFull, {}, fill_unspecified_inner_fields_with_nulls=True
    )
    expected = create_partially_filled_dataset(
        spark,
        TopStructDeepFull,
        {
            TopStructDeepFull.id: [1, 2],
            TopStructDeepFull.wrapper: [
                {"n": 1, "leaf": {"x": 10, "y": None}},
                {"n": 2, "leaf": {"x": 20, "y": None}},
            ],
        },
    )
    assert_df_equality(observed, expected)


def test_fill_unspecified_inner_fields_missing_sub_struct(spark: SparkSession):
    class Sub(Schema):
        p: Column[IntegerType]

    class InnerWithSub(Schema):
        x: Column[IntegerType]
        sub: Column[StructType[Sub]]

    class TopStructSubFull(Schema):
        id: Column[IntegerType]
        inner: Column[StructType[InnerWithSub]]

    ds = create_partially_filled_dataset(
        spark,
        TopStructPartial,
        {TopStructPartial.id: [1, 2], TopStructPartial.inner: [{"x": 10}, {"x": 20}]},
    )
    observed = transform_to_schema(
        ds, TopStructSubFull, {}, fill_unspecified_inner_fields_with_nulls=True
    )
    expected = create_partially_filled_dataset(
        spark,
        TopStructSubFull,
        {
            TopStructSubFull.id: [1, 2],
            TopStructSubFull.inner: [{"x": 10, "sub": None}, {"x": 20, "sub": None}],
        },
    )
    assert_df_equality(observed, expected)


def test_fill_unspecified_inner_fields_in_array(spark: SparkSession):
    class TopArrayFull(Schema):
        id: Column[IntegerType]
        items: Column[ArrayType[StructType[InnerFull]]]

    class TopArrayPartial(Schema):
        id: Column[IntegerType]
        items: Column[ArrayType[StructType[InnerPartial]]]

    ds = create_partially_filled_dataset(
        spark,
        TopArrayPartial,
        {
            TopArrayPartial.id: [1, 2],
            TopArrayPartial.items: [[{"x": 10}, {"x": 11}], [{"x": 20}]],
        },
    )
    observed = transform_to_schema(
        ds, TopArrayFull, {}, fill_unspecified_inner_fields_with_nulls=True
    )
    expected = create_partially_filled_dataset(
        spark,
        TopArrayFull,
        {
            TopArrayFull.id: [1, 2],
            TopArrayFull.items: [
                [{"x": 10, "y": None}, {"x": 11, "y": None}],
                [{"x": 20, "y": None}],
            ],
        },
    )
    assert_df_equality(observed, expected)


def test_fill_unspecified_inner_fields_in_map(spark: SparkSession):
    class TopMapFull(Schema):
        id: Column[IntegerType]
        mapping: Column[MapType[StringType, StructType[InnerFull]]]

    class TopMapPartial(Schema):
        id: Column[IntegerType]
        mapping: Column[MapType[StringType, StructType[InnerPartial]]]

    ds = create_partially_filled_dataset(
        spark,
        TopMapPartial,
        {
            TopMapPartial.id: [1, 2],
            TopMapPartial.mapping: [{"a": {"x": 10}}, {"b": {"x": 20}}],
        },
    )
    observed = transform_to_schema(
        ds, TopMapFull, {}, fill_unspecified_inner_fields_with_nulls=True
    )
    expected = create_partially_filled_dataset(
        spark,
        TopMapFull,
        {
            TopMapFull.id: [1, 2],
            TopMapFull.mapping: [{"a": {"x": 10, "y": None}}, {"b": {"x": 20, "y": None}}],
        },
    )
    assert_df_equality(observed, expected)


def test_fill_unspecified_inner_fields_skips_explicitly_transformed_columns(spark: SparkSession):
    """Columns already in transformations should be skipped by auto-fill."""
    from pyspark.sql import functions as F

    class TopStructFull(Schema):
        id: Column[IntegerType]
        inner: Column[StructType[InnerFull]]

    ds = create_partially_filled_dataset(
        spark,
        TopStructPartial,
        {TopStructPartial.id: [1, 2], TopStructPartial.inner: [{"x": 10}, {"x": 20}]},
    )
    observed = transform_to_schema(
        ds,
        TopStructFull,
        {TopStructFull.inner: F.struct(F.lit(99).alias("x"), F.lit(42).alias("y"))},
        fill_unspecified_inner_fields_with_nulls=True,
    )
    expected = create_partially_filled_dataset(
        spark,
        TopStructFull,
        {
            TopStructFull.id: [1, 2],
            TopStructFull.inner: [{"x": 99, "y": 42}, {"x": 99, "y": 42}],
        },
    )
    assert_df_equality(observed, expected, ignore_nullable=True)


def test_fill_unspecified_inner_fields_skips_columns_absent_from_data(spark: SparkSession):
    """fill_unspecified_inner_fields_with_nulls should skips top-level columns absent
    from the data.

    Without fill_unspecified_columns_with_nulls the missing column should cause an
    error.
    """

    class IdOnly(Schema):
        id: Column[IntegerType]

    class TopStructFull(Schema):
        id: Column[IntegerType]
        inner: Column[StructType[InnerFull]]

    ds = create_partially_filled_dataset(spark, IdOnly, {IdOnly.id: [1, 2]})
    with pytest.raises(AnalysisException):
        transform_to_schema(
            ds,
            TopStructFull,
            {},
            fill_unspecified_inner_fields_with_nulls=True,
        )


def test_fill_unspecified_inner_fields_map_key_type_mismatch_raises(spark: SparkSession):
    """Map key type mismatch should raises TypeError."""

    class TopMapIntKey(Schema):
        id: Column[IntegerType]
        mapping: Column[MapType[IntegerType, StructType[InnerFull]]]

    class TopMapStrKey(Schema):
        id: Column[IntegerType]
        mapping: Column[MapType[StringType, StructType[InnerFull]]]

    ds = create_partially_filled_dataset(
        spark,
        TopMapIntKey,
        {TopMapIntKey.id: [1], TopMapIntKey.mapping: [{1: {"x": 10, "y": 20}}]},
    )
    with pytest.raises(TypeError):
        transform_to_schema(ds, TopMapStrKey, {}, fill_unspecified_inner_fields_with_nulls=True)
