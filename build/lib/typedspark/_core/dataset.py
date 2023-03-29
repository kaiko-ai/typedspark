"""Module containing classes and functions related to TypedSpark DataSets."""
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    overload,
)

from pyspark.sql import Column as SparkColumn
from pyspark.sql import DataFrame
from typing_extensions import Concatenate, ParamSpec

from typedspark._core.validate_schema import validate_schema
from typedspark._schema.schema import Schema

T = TypeVar("T", bound=Schema)
_ReturnType = TypeVar("_ReturnType", bound=DataFrame)  # pylint: disable=C0103
P = ParamSpec("P")


class DataSet(DataFrame, Generic[T]):
    """TypedSpark DataSet."""

    def __new__(cls, dataframe: DataFrame) -> "DataSet[T]":
        """`__new__()` instantiates the object (prior to `__init__()`).

        Here, we simply take the provided `df` and cast it to a
        `DataSet`. This allows us to bypass the `DataFrame` constuctor
        in `__init__()`, which requires parameters that may be difficult
        to access.
        """
        dataframe.__class__ = DataSet
        return dataframe  # type: ignore

    def __init__(self, dataframe: DataFrame):
        pass

    def __setattr__(self, name: str, value: Any) -> None:
        """Python base function that sets attributes.

        We listen here for the setting of `__orig_class__`, which
        contains the `Schema` of the `DataSet`. Note that this gets set
        after `__new__()` and `__init__()` are finished.
        """
        object.__setattr__(self, name, value)

        if name == "__orig_class__":
            orig_class_args = get_args(self.__orig_class__)
            if orig_class_args and issubclass(orig_class_args[0], Schema):
                self._schema_annotations: Type[Schema] = get_args(value)[0]
                validate_schema(
                    self._schema_annotations.get_structtype(),
                    deepcopy(self.schema),
                    self._schema_annotations.get_schema_name(),
                )
                self._add_schema_metadata()

    def _add_schema_metadata(self) -> None:
        """Adds the `ColumnMeta` comments as metadata to the `DataSet`.

        Previously set metadata is deleted. Hence, if `foo(dataframe: DataSet[A]) -> DataSet[B]`,
        then `DataSet[B]` will not inherrit any metadata from `DataSet[A]`.

        Assumes validate_schema() in __setattr__() has been run.
        """
        for field in self._schema_annotations.get_structtype().fields:
            self.schema[field.name].metadata = field.metadata

    """The following functions are equivalent to their parents in `DataFrame`, but since they
    don't affect the `Schema`, we can add type annotations here. We're omitting docstrings,
    such that the docstring from the parent will appear."""

    def distinct(self) -> "DataSet[T]":  # pylint: disable=C0116
        return DataSet[self._schema_annotations](super().distinct())  # type: ignore

    def filter(self, condition) -> "DataSet[T]":  # pylint: disable=C0116
        return DataSet[self._schema_annotations](super().filter(condition))  # type: ignore

    @overload
    def intersect(self, other: "DataSet[T]") -> "DataSet[T]":
        ...  # pragma: no cover

    @overload
    def intersect(self, other: DataFrame) -> DataFrame:
        ...  # pragma: no cover

    def intersect(self, other: DataFrame) -> DataFrame:  # pylint: disable=C0116
        res = super().intersect(other)
        if isinstance(other, DataSet) and other._schema_annotations == self._schema_annotations:
            return DataSet[self._schema_annotations](res)  # type: ignore
        return res  # pragma: no cover

    @overload
    def join(
        self,
        other: DataFrame,
        on: Optional[  # pylint: disable=C0103
            Union[str, List[str], SparkColumn, List[SparkColumn]]
        ] = None,
        how: Literal["semi"] = "semi",
    ) -> "DataSet[T]":
        ...  # pragma: no cover

    @overload
    def join(
        self,
        other: DataFrame,
        on: Optional[  # pylint: disable=C0103
            Union[str, List[str], SparkColumn, List[SparkColumn]]
        ] = None,
        how: Optional[str] = None,
    ) -> DataFrame:
        ...  # pragma: no cover

    def join(  # pylint: disable=C0116
        self,
        other: DataFrame,
        on: Optional[  # pylint: disable=C0103
            Union[str, List[str], SparkColumn, List[SparkColumn]]
        ] = None,
        how: Optional[str] = None,
    ) -> DataFrame:
        return super().join(other, on, how)  # type: ignore

    def orderBy(self, *args, **kwargs) -> "DataSet[T]":  # type: ignore  # noqa: N802, E501  # pylint: disable=C0116, C0103
        return DataSet[self._schema_annotations](super().orderBy(*args, **kwargs))  # type: ignore

    @overload
    def transform(
        self,
        func: Callable[Concatenate["DataSet[T]", P], _ReturnType],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> _ReturnType:
        ...  # pragma: no cover

    @overload
    def transform(self, func: Callable[..., "DataFrame"], *args: Any, **kwargs: Any) -> "DataFrame":
        ...  # pragma: no cover

    def transform(  # pylint: disable=C0116
        self, func: Callable[..., "DataFrame"], *args: Any, **kwargs: Any
    ) -> "DataFrame":
        return super().transform(func, *args, **kwargs)

    @overload
    def union(self, other: "DataSet[T]") -> "DataSet[T]":
        ...  # pragma: no cover

    @overload
    def union(self, other: DataFrame) -> DataFrame:
        ...  # pragma: no cover

    def union(self, other: DataFrame) -> DataFrame:  # pylint: disable=C0116
        res = super().union(other)
        if isinstance(other, DataSet) and other._schema_annotations == self._schema_annotations:
            return DataSet[self._schema_annotations](res)  # type: ignore
        return res  # pragma: no cover

    @overload
    def unionByName(  # noqa: N802  # pylint: disable=C0116, C0103
        self,
        other: "DataSet[T]",
        allowMissingColumns: Literal[False] = False,  # noqa: N803
    ) -> "DataSet[T]":
        ...  # pragma: no cover

    @overload
    def unionByName(  # noqa: N802  # pylint: disable=C0116, C0103
        self,
        other: DataFrame,
        allowMissingColumns: bool = False,  # noqa: N803
    ) -> DataFrame:
        ...  # pragma: no cover

    def unionByName(  # noqa: N802  # pylint: disable=C0116, C0103
        self,
        other: DataFrame,
        allowMissingColumns: bool = False,  # noqa: N803
    ) -> DataFrame:
        res = super().unionByName(other, allowMissingColumns)
        if isinstance(other, DataSet) and other._schema_annotations == self._schema_annotations:
            return DataSet[self._schema_annotations](res)  # type: ignore
        return res  # pragma: no cover
