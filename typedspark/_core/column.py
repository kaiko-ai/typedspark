"""Module containing classes and functions related to TypedSpark Columns."""

from logging import warn
from typing import Generic, Optional, TypeVar, Union, get_args, get_origin

from pyspark.sql import Column as SparkColumn
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DataType

from typedspark._core.datatypes import StructType

T = TypeVar("T", bound=DataType)


class EmptyColumn(SparkColumn):
    """Column object to be instantiated when there is no active Spark session."""

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover
        pass


class Column(SparkColumn, Generic[T]):
    """Represents a ``Column`` in a ``Schema``. Can be used as:

    .. code-block:: python

        class A(Schema):
            a: Column[IntegerType]
            b: Column[StringType]
    """

    def __new__(
        cls,
        name: str,
        dataframe: Optional[DataFrame] = None,
        curid: Optional[int] = None,
        dtype: Optional[T] = None,
        parent: Union[DataFrame, "Column", None] = None,
        alias: Optional[str] = None,
    ):
        """``__new__()`` instantiates the object (prior to ``__init__()``).

        Here, we simply take the provided ``name``, create a pyspark
        ``Column`` object and cast it to a typedspark ``Column`` object.
        This allows us to bypass the pypsark ``Column`` constuctor in
        ``__init__()``, which requires parameters that may be difficult
        to access.
        """
        # pylint: disable=unused-argument

        if dataframe is not None and parent is None:
            parent = dataframe
            warn("The use of Column(dataframe=...) is deprecated, use Column(parent=...) instead.")

        column: SparkColumn
        if SparkSession.getActiveSession() is None:
            column = EmptyColumn()  # pragma: no cover
        elif alias is not None:
            column = col(f"{alias}.{name}")
        elif parent is not None:
            column = parent[name]
        else:
            column = col(name)

        column.__class__ = Column  # type: ignore
        return column

    def __init__(
        self,
        name: str,
        dataframe: Optional[DataFrame] = None,
        curid: Optional[int] = None,
        dtype: Optional[T] = None,
        parent: Union[DataFrame, "Column", None] = None,
        alias: Optional[str] = None,
    ):
        # pylint: disable=unused-argument
        self.str = name
        self._dtype = dtype if dtype is not None else DataType
        self._curid = curid

    def __hash__(self) -> int:
        return hash((self.str, self._curid))

    @property
    def dtype(self) -> T:
        """Get the datatype of the column, e.g. Column[IntegerType] -> IntegerType."""
        dtype = self._dtype

        if get_origin(dtype) == StructType:
            return StructType(
                schema=get_args(dtype)[0],
                parent=self,
            )  # type: ignore

        return dtype()  # type: ignore
