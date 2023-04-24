"""Module containing classes and functions related to TypedSpark Columns."""

from typing import Generic, Optional, Type, TypeVar, get_type_hints

from pyspark.sql import Column as SparkColumn
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DataType

from typedspark._core.utils import get_dtype_from_column, get_schema_from_structtype, is_structtype

T = TypeVar("T", bound=DataType)


class EmptyColumn(SparkColumn):
    """Column object to be instantiated when there is no active Spark
    session."""

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
        dtype: Optional[Type[DataType]] = None,
    ):
        """``__new__()`` instantiates the object (prior to ``__init__()``).

        Here, we simply take the provided ``name``, create a pyspark
        ``Column`` object and cast it to a typedspark ``Column`` object.
        This allows us to bypass the pypsark ``Column`` constuctor in
        ``__init__()``, which requires parameters that may be difficult
        to access.
        """
        # pylint: disable=unused-argument

        column: SparkColumn
        if SparkSession.getActiveSession() is None:
            column = EmptyColumn()  # pragma: no cover
        elif dataframe is None:
            column = col(name)
        else:
            column = dataframe[name]

        column.__class__ = Column
        return column

    def __init__(
        self,
        name: str,
        dataframe: Optional[DataFrame] = None,
        curid: Optional[int] = None,
        dtype: Optional[Type[DataType]] = None,
    ):
        # pylint: disable=unused-argument
        self.str = name
        self._curid = curid

        if dtype and is_structtype(dtype):
            self._set_structtype_attributes(dtype)

    def _set_structtype_attributes(self, dtype: Type[DataType]) -> None:
        schema = get_schema_from_structtype(dtype)
        for name, column in get_type_hints(schema).items():
            self.__setattr__(name, Column(name, dtype=get_dtype_from_column(column)))

    def __hash__(self) -> int:
        return hash((self.str, self._curid))
