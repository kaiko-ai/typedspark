"""Module containing classes and functions related to TypedSpark DataSets."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Generic, List, Literal, Optional, Type, TypeVar, Union, cast, overload

from pyspark import StorageLevel
from pyspark.sql import Column as SparkColumn
from pyspark.sql import DataFrame
from typing_extensions import Concatenate, ParamSpec

from typedspark._core.validate_schema import validate_schema
from typedspark._schema.schema import Schema

_Schema = TypeVar("_Schema", bound=Schema)
_Protocol = TypeVar("_Protocol", bound=Schema, covariant=True)
_Implementation = TypeVar("_Implementation", bound=Schema, covariant=True)

P = ParamSpec("P")
_ReturnType = TypeVar("_ReturnType", bound=DataFrame)  # pylint: disable=C0103


class DataSetImplements(DataFrame, Generic[_Protocol, _Implementation]):
    """DataSetImplements allows us to define functions such as:

    .. code-block:: python
        class Age(Schema, Protocol):
            age: Column[LongType]

        def birthday(df: DataSetImplements[Age, T]) -> DataSet[T]:
            return transform_to_schema(
                df,
                df.typedspark_schema,
                {Age.age: Age.age + 1},
            )

    Such a function:
    1. Takes as an input ``DataSetImplements[Age, T]``: a ``DataSet`` that implements the protocol
       ``Age`` as ``T``.
    2. Returns a ``DataSet[T]``: a ``DataSet`` of the same type as the one that was provided.

    ``DataSetImplements`` should solely be used as a type annotation, it is never initialized."""

    _schema_annotations: Type[_Implementation]

    def __init__(self):
        raise NotImplementedError(
            "DataSetImplements should solely be used as a type annotation, it is never initialized."
        )

    @property
    def typedspark_schema(self) -> Type[_Implementation]:
        """Returns the ``Schema`` of the ``DataSet``."""
        return self._schema_annotations

    """The following functions are equivalent to their parents in ``DataFrame``, but since they
    don't affect the ``Schema``, we can add type annotations here. We're omitting docstrings,
    such that the docstring from the parent will appear."""

    def alias(self, alias: str) -> DataSet[_Implementation]:
        return DataSet[self._schema_annotations](super().alias(alias))  # type: ignore

    def cache(self) -> DataSet[_Implementation]:  # pylint: disable=C0116
        return DataSet[self._schema_annotations](super().cache())  # type: ignore

    def persist(
        self,
        storageLevel: StorageLevel = (StorageLevel.MEMORY_AND_DISK_DESER),
    ) -> DataSet[_Implementation]:
        return DataSet[self._schema_annotations](super().persist(storageLevel))  # type: ignore

    def unpersist(self, blocking: bool = False) -> DataSet[_Implementation]:
        return DataSet[self._schema_annotations](super().unpersist(blocking))  # type: ignore

    def distinct(self) -> DataSet[_Implementation]:  # pylint: disable=C0116
        return DataSet[self._schema_annotations](super().distinct())  # type: ignore

    def filter(self, condition) -> DataSet[_Implementation]:  # type: ignore[override]
        """Filters rows using the given condition"""
        return DataSet[self._schema_annotations](super().filter(condition))  # type: ignore

    def where(self, condition) -> DataSet[_Implementation]:  # type: ignore[override]
        """Filters rows using the given condition"""
        return DataSet[self._schema_annotations](super().where(condition))  # type: ignore

    @overload
    def join(  # type: ignore
        self,
        other: DataFrame,
        on: Optional[  # pylint: disable=C0103
            Union[str, List[str], SparkColumn, List[SparkColumn]]
        ] = ...,
        how: None = ...,
    ) -> DataFrame: ...  # pragma: no cover

    @overload
    def join(
        self,
        other: DataFrame,
        on: Optional[  # pylint: disable=C0103
            Union[str, List[str], SparkColumn, List[SparkColumn]]
        ] = ...,
        how: Literal["semi"] = ...,
    ) -> DataSet[_Implementation]: ...  # pragma: no cover

    @overload
    def join(
        self,
        other: DataFrame,
        on: Optional[  # pylint: disable=C0103
            Union[str, List[str], SparkColumn, List[SparkColumn]]
        ] = ...,
        how: Optional[str] = ...,
    ) -> DataFrame: ...  # pragma: no cover

    def join(  # pylint: disable=C0116
        self,
        other: DataFrame,
        on: Optional[  # pylint: disable=C0103
            Union[str, List[str], SparkColumn, List[SparkColumn]]
        ] = None,
        how: Optional[str] = None,
    ) -> DataFrame:
        return super().join(other, on, how)  # type: ignore

    def orderBy(self, *args, **kwargs) -> DataSet[_Implementation]:  # type: ignore  # noqa: N802, E501  # pylint: disable=C0116, C0103
        return DataSet[self._schema_annotations](super().orderBy(*args, **kwargs))  # type: ignore

    def transform(
        self,
        func: Callable[Concatenate[DataSet[_Implementation], P], _ReturnType],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> _ReturnType:
        return super().transform(func, *args, **kwargs)  # type: ignore

    @overload
    def unionByName(  # noqa: N802  # pylint: disable=C0116, C0103
        self,
        other: DataSet[_Implementation],
        allowMissingColumns: Literal[False] = ...,  # noqa: N803
    ) -> DataSet[_Implementation]: ...  # pragma: no cover

    @overload
    def unionByName(  # noqa: N802  # pylint: disable=C0116, C0103
        self,
        other: DataFrame,
        allowMissingColumns: bool = ...,  # noqa: N803
    ) -> DataFrame: ...  # pragma: no cover

    def unionByName(  # noqa: N802  # pylint: disable=C0116, C0103
        self,
        other: DataFrame,
        allowMissingColumns: bool = False,  # noqa: N803
    ) -> DataFrame:
        res = super().unionByName(other, allowMissingColumns)
        if isinstance(other, DataSet) and other._schema_annotations == self._schema_annotations:
            return DataSet[self._schema_annotations](res)  # type: ignore
        return res  # pragma: no cover


class DataSet(DataSetImplements[_Schema, _Schema]):
    """``DataSet`` subclasses pyspark ``DataFrame`` and hence has all the same
    functionality, with in addition the possibility to define a schema.

    .. code-block:: python

        class Person(Schema):
            name: Column[StringType]
            age: Column[LongType]

        def foo(df: DataSet[Person]) -> DataSet[Person]:
            # do stuff
            return df
    """

    def __new__(cls, dataframe: DataFrame) -> DataSet[_Schema]:
        """``__new__()`` instantiates the object (prior to ``__init__()``).

        Here, we simply take the provided ``df`` and cast it to a
        ``DataSet``. This allows us to bypass the ``DataFrame``
        constuctor in ``__init__()``, which requires parameters that may
        be difficult to access. Subsequently, we perform schema validation, if
        the schema annotations are provided.
        """
        dataframe = cast(DataSet, dataframe)
        dataframe.__class__ = DataSet

        # first we reset the schema annotations to None, in case they are inherrited through the
        # passed DataFrame
        dataframe._schema_annotations = None  # type: ignore

        # then we use the class' schema annotations to validate the schema and add metadata
        if hasattr(cls, "_schema_annotations"):
            dataframe._schema_annotations = cls._schema_annotations  # type: ignore
            dataframe._validate_schema()
            dataframe._add_schema_metadata()

        return dataframe  # type: ignore

    def __init__(self, dataframe: DataFrame):
        pass

    def __class_getitem__(cls, item):
        """Allows us to define a schema for the ``DataSet``.

        To make sure that the DataSet._schema_annotations variable isn't reused globally, we
        generate a subclass of the ``DataSet`` with the schema annotations as a class variable.
        """
        subclass_name = f"{cls.__name__}[{item.__name__}]"
        subclass = type(subclass_name, (cls,), {"_schema_annotations": item})
        return subclass

    def _validate_schema(self) -> None:
        """Validates the schema of the ``DataSet`` against the schema annotations."""
        validate_schema(
            self._schema_annotations.get_structtype(),
            deepcopy(self.schema),
            self._schema_annotations.get_schema_name(),
        )

    def _add_schema_metadata(self) -> None:
        """Adds the ``ColumnMeta`` comments as metadata to the ``DataSet``.

        Previously set metadata is deleted. Hence, if ``foo(dataframe: DataSet[A]) -> DataSet[B]``,
        then ``DataSet[B]`` will not inherrit any metadata from ``DataSet[A]``.

        Assumes validate_schema() in __setattr__() has been run.
        """
        for field in self._schema_annotations.get_structtype().fields:
            self.schema[field.name].metadata = field.metadata

    """The following functions are equivalent to their parents in ``DataSetImplements``. However,
    to support functions like ``functools.reduce(DataSet.unionByName, datasets)``, we also add them
    here. Unfortunately, this leads to some code redundancy, but we'll take that for granted."""

    def alias(self, alias: str) -> DataSet[_Schema]:
        return DataSet[self._schema_annotations](super().alias(alias))  # type: ignore

    def cache(self) -> DataSet[_Schema]:  # pylint: disable=C0116
        return DataSet[self._schema_annotations](super().cache())  # type: ignore

    def persist(
        self,
        storageLevel: StorageLevel = (StorageLevel.MEMORY_AND_DISK_DESER),
    ) -> DataSet[_Schema]:  # pylint: disable=C0116
        return DataSet[self._schema_annotations](super().persist(storageLevel))  # type: ignore

    def unpersist(self, blocking: bool = False) -> DataSet[_Schema]:  # pylint: disable=C0116
        return DataSet[self._schema_annotations](super().unpersist(blocking))  # type: ignore

    def distinct(self) -> DataSet[_Schema]:  # pylint: disable=C0116
        return DataSet[self._schema_annotations](super().distinct())  # type: ignore

    def filter(self, condition) -> DataSet[_Schema]:  # type: ignore[override]
        """Filters rows using the given condition"""
        return DataSet[self._schema_annotations](super().filter(condition))  # type: ignore

    def where(self, condition) -> DataSet[_Schema]:  # type: ignore[override]
        """Filters rows using the given condition"""
        return DataSet[self._schema_annotations](super().where(condition))  # type: ignore

    @overload
    def join(  # type: ignore
        self,
        other: DataFrame,
        on: Optional[  # pylint: disable=C0103
            Union[str, List[str], SparkColumn, List[SparkColumn]]
        ] = ...,
        how: None = ...,
    ) -> DataFrame: ...  # pragma: no cover

    @overload
    def join(
        self,
        other: DataFrame,
        on: Optional[  # pylint: disable=C0103
            Union[str, List[str], SparkColumn, List[SparkColumn]]
        ] = ...,
        how: Literal["semi"] = ...,
    ) -> DataSet[_Schema]: ...  # pragma: no cover

    @overload
    def join(
        self,
        other: DataFrame,
        on: Optional[  # pylint: disable=C0103
            Union[str, List[str], SparkColumn, List[SparkColumn]]
        ] = ...,
        how: Optional[str] = ...,
    ) -> DataFrame: ...  # pragma: no cover

    def join(  # pylint: disable=C0116
        self,
        other: DataFrame,
        on: Optional[  # pylint: disable=C0103
            Union[str, List[str], SparkColumn, List[SparkColumn]]
        ] = None,
        how: Optional[str] = None,
    ) -> DataFrame:
        return super().join(other, on, how)  # type: ignore

    def orderBy(self, *args, **kwargs) -> DataSet[_Schema]:  # type: ignore  # noqa: N802, E501  # pylint: disable=C0116, C0103
        return DataSet[self._schema_annotations](super().orderBy(*args, **kwargs))  # type: ignore

    def transform(
        self,
        func: Callable[Concatenate[DataSet[_Schema], P], _ReturnType],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> _ReturnType:
        return super().transform(func, *args, **kwargs)  # type: ignore

    @overload
    def unionByName(  # noqa: N802  # pylint: disable=C0116, C0103
        self,
        other: DataSet[_Schema],
        allowMissingColumns: Literal[False] = ...,  # noqa: N803
    ) -> DataSet[_Schema]: ...  # pragma: no cover

    @overload
    def unionByName(  # noqa: N802  # pylint: disable=C0116, C0103
        self,
        other: DataFrame,
        allowMissingColumns: bool = ...,  # noqa: N803
    ) -> DataFrame: ...  # pragma: no cover

    def unionByName(  # noqa: N802  # pylint: disable=C0116, C0103
        self,
        other: DataFrame,
        allowMissingColumns: bool = False,  # noqa: N803
    ) -> DataFrame:
        res = super().unionByName(other, allowMissingColumns)
        if isinstance(other, DataSet) and other._schema_annotations == self._schema_annotations:
            return DataSet[self._schema_annotations](res)  # type: ignore
        return res  # pragma: no cover
