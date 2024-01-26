"""Module containing functions that are related to registering schema's to DataSets."""

import itertools
from typing import Tuple, Type, TypeVar

from typedspark._core.dataset import DataSet
from typedspark._schema.schema import Schema

T = TypeVar("T", bound=Schema)


def _counter(count: itertools.count = itertools.count()):
    return next(count)


def register_schema_to_dataset(dataframe: DataSet[T], schema: Type[T]) -> Type[T]:
    """Helps combat column ambiguity. For example:

    .. code-block:: python

        class Person(Schema):
            id: Column[IntegerType]
            name: Column[StringType]

        class Job(Schema):
            id: Column[IntegerType]
            salary: Column[IntegerType]

        class PersonWithJob(Person, Job):
            pass

        def foo(df_a: DataSet[Person], df_b: DataSet[Job]) -> DataSet[PersonWithJob]:
            return DataSet[PersonWithSalary](
                df_a.join(
                    df_b,
                    Person.id == Job.id
                )
            )

    Calling ``foo()`` would result in a ``AnalysisException``, because Spark can't figure out
    whether ``id`` belongs to ``df_a`` or ``df_b``. To deal with this, you need to register
    your ``Schema`` to the ``DataSet``.

    .. code-block:: python

        from typedspark import register_schema_to_dataset

        def foo(df_a: DataSet[Person], df_b: DataSet[Job]) -> DataSet[PersonWithSalary]:
            person = register_schema_to_dataset(df_a, Person)
            job = register_schema_to_dataset(df_b, Job)
            return DataSet[PersonWithSalary](
                df_a.join(
                    df_b,
                    person.id == job.id
                )
            )
    """

    class LinkedSchema(schema):  # type: ignore  # pylint: disable=missing-class-docstring
        _parent = dataframe
        _current_id = _counter()
        _original_name = schema.get_schema_name()

    return LinkedSchema  # type: ignore


def register_schema_to_dataset_with_alias(
    dataframe: DataSet[T], schema: Type[T], alias: str
) -> Tuple[DataSet[T], Type[T]]:
    """When dealing with self-joins, running `register_dataset_to_schema()` is not
    enough.

    Instead, we'll need `register_dataset_to_schema_with_alias()`, e.g.:

    .. code-block:: python

        class Person(Schema):
            id: Column[IntegerType]
            name: Column[StringType]

        df_a, person_a = register_schema_to_dataset_with_alias(df, Person, alias="a")
        df_b, person_b = register_schema_to_dataset_with_alias(df, Person, alias="b")

        df_a.join(df_b, person_a.id == person_b.id)
    """

    class LinkedSchema(schema):  # type: ignore  # pylint: disable=missing-class-docstring
        _current_id = _counter()
        _original_name = schema.get_schema_name()
        _alias = alias

    return (
        dataframe.alias(alias),
        LinkedSchema,  # type: ignore
    )
