"""Module containing classes and functions related to TypedSpark Columns."""

from typing import Generic, Optional, TypeVar, get_args, Any

from typedspark._core.datatypes import StructType

from pyspark.sql import Column as SparkColumn
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DataType

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
    ):
        # pylint: disable=unused-argument
        self.str = name
        self._curid = curid

    def __setattr__(self, name: str, value: Any) -> None:
        """Python base function that sets attributes.

        We listen here for the setting of ``__orig_class__``, which
        contains the type of the column. Note that this gets
        set after ``__new__()`` and ``__init__()`` are finished.
        """
        object.__setattr__(self, name, value)

        if name == "__orig_class__":
            orig_class_args = get_args(value)
            if not orig_class_args or orig_class_args[0] != StructType:
                return
            
            structtype_args = get_args(orig_class_args[0])
            if not structtype_args:  # or not issubclass(structtype_args[0], Schema)
                return
            
            schema = structtype_args[0]
            for field in schema.get_structtype().fields:
                self.__setattr__(field.name, self.__getattribute__(field.name))

    def __hash__(self) -> int:
        return hash((self.str, self._curid))
