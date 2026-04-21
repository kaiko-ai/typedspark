"""Module containing classes and functions related to TypedSpark Columns."""

from logging import warning
from typing import Generic, Optional, TypeVar, Union, get_args, get_origin

from pyspark.errors import PySparkRuntimeError
from pyspark.sql import Column as SparkColumn
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DataType

from typedspark._core.datatypes import StructType
from typedspark._utils.pyspark_compat import attach_mixin

T = TypeVar("T", bound=DataType)


class EmptyColumn(SparkColumn):
    """Column object to be instantiated when there is no active Spark session."""

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover
        pass


def _get_active_or_default_session() -> Optional[SparkSession]:
    """Return the active Spark session, falling back to the default/instantiated
    session."""
    # SparkSession.active() relies on getActiveSession(), falls back to _instantiatedSession,
    # and raises PySparkRuntimeError("NO_ACTIVE_OR_DEFAULT_SESSION") when neither exists.
    try:
        return SparkSession.active()
    except PySparkRuntimeError as exc:
        if exc.getErrorClass() == "NO_ACTIVE_OR_DEFAULT_SESSION":
            return None

        raise


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
            warning(
                "The use of Column(dataframe=...) is deprecated, use Column(parent=...) instead."
            )

        column: SparkColumn
        if _get_active_or_default_session() is None:
            column = EmptyColumn()  # pragma: no cover
        elif alias is not None:
            column = col(f"{alias}.{name}")
        elif parent is not None:
            column = parent[name]
        else:
            column = col(name)

        attach_mixin(column, cls)
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
        self._parent = parent

    def __hash__(self) -> int:
        return hash((self.str, self._curid))

    @property
    def full_path(self) -> str:
        """Full path of the column including parent structure.
        Example:
        .. code-block:: python
            from pyspark.sql.types import IntegerType, StringType
            from typedspark import DataSet, StructType, Schema, Column

            class Values(Schema):
                name: Column[StringType]
                severity: Column[IntegerType]


            class Actions(Schema):
                consequences: Column[StructType[Values]]

        `Actions.consequences.dtype.schema.severity.full_path` will yield the name
        of the field `severity` including the full path: `consequences.severity`

        """
        if isinstance(self._parent, Column):
            return f"{self._parent.full_path}.{self.str}"
        return self.str

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

    def __repr__(self) -> str:
        spark = _get_active_or_default_session()
        if spark is None or not hasattr(self, "_jc"):  # pragma: no cover
            # Columns created without a session stay "empty" even if a session appears later.
            return f"Column<'{self.str}'> (no active Spark session)"

        return super().__repr__()
