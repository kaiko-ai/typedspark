"""Loads all catalogs, databases and tables in a SparkSession."""

from abc import ABC
from datetime import datetime
from typing import Any, Optional, Tuple, TypeVar
from warnings import warn

from pyspark.sql import Row, SparkSession

from typedspark._core.dataset import DataSet
from typedspark._schema.schema import Schema
from typedspark._utils.camelcase import to_camel_case
from typedspark._utils.load_table import load_table

T = TypeVar("T", bound=Schema)


class Timeout(ABC):
    """Warns the user if loading databases or catalogs is taking too long."""

    _TIMEOUT_WARNING: str

    def __init__(self, silent: bool, n: int):  # pylint: disable=invalid-name
        self._start = datetime.now()
        self._silent = silent
        self._n = n

    def check_for_warning(self, i: int):  # pragma: no cover
        """Checks if a warning should be issued."""
        if self._silent:
            return

        diff = datetime.now() - self._start
        if diff.seconds > 10:
            warn(self._TIMEOUT_WARNING.format(i, self._n))
            self._silent = True


class DatabasesTimeout(Timeout):
    """Warns the user if Databases() is taking too long."""

    _TIMEOUT_WARNING = """
Databases() is taking longer than 10 seconds. So far, {} out of {} databases have been loaded.
If this is too slow, consider loading a single database using:

from typedspark import Database

db = Database(spark, db_name=...)
"""


class CatalogsTimeout(Timeout):
    """Warns the user if Catalogs() is taking too long."""

    _TIMEOUT_WARNING = """
Catalogs() is taking longer than 10 seconds. So far, {} out of {} catalogs have been loaded.
If this is too slow, consider loading a single catalog using:

from typedspark import Databases

db = Databases(spark, catalog_name=...)
"""


def _get_spark_session(spark: Optional[SparkSession]) -> SparkSession:
    if spark is not None:
        return spark

    spark = SparkSession.getActiveSession()
    if spark is not None:
        return spark

    raise ValueError("No active SparkSession found.")  # pragma: no cover


def _resolve_names_starting_with_an_underscore(name: str, names: list[str]) -> str:
    """Autocomplete is currently problematic when a name (of a table, database, or
    catlog) starts with an underscore.

    In this case, it's considered a private attribute and it doesn't show up in the
    autocomplete options in your notebook. To combat this behaviour, we add a u as a
    prefix, followed by as many underscores as needed (up to 100) to keep the name
    unique.
    """
    if not name.startswith("_"):
        return name

    prefix = "u"
    proposed_name = prefix + name
    i = 0
    while proposed_name in names:
        prefix = prefix + "_"
        proposed_name = prefix + name
        i += 1
        if i > 100:
            raise Exception(
                "Couldn't find a unique name, even when adding 100 underscores. This seems unlikely"
                " behaviour, exiting to prevent an infinite loop."
            )  # pragma: no cover

    return proposed_name


class Table:
    """Loads a table in a database."""

    def __init__(self, spark: SparkSession, db_name: str, table_name: str, is_temporary: bool):
        self._spark = spark
        self._db_name = db_name
        self._table_name = table_name
        self._is_temporary = is_temporary

    @property
    def str(self) -> str:
        """Returns the path to the table, e.g. ``default.person``.

        While temporary tables are always stored in the ``default`` db, they are saved and
        loaded directly from their table name, e.g. ``person``.

        Non-temporary tables are saved and loaded from their full name, e.g.
        ``default.person``.
        """
        if self._is_temporary:
            return self._table_name

        return f"{self._db_name}.{self._table_name}"

    def load(self) -> Tuple[DataSet[T], T]:
        """Loads the table as a DataSet[T] and returns the schema."""
        return load_table(  # type: ignore
            self._spark,
            self.str,
            to_camel_case(self._table_name),
        )

    def __call__(self, *args: Any, **kwds: Any) -> Tuple[DataSet[T], T]:
        return self.load()


class Database:
    """Loads all tables in a database."""

    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        db_name: str = "default",
        catalog_name: Optional[str] = None,
    ):
        spark = _get_spark_session(spark)

        if catalog_name is None:
            self._db_name = db_name
        else:
            self._db_name = f"{catalog_name}.{db_name}"

        tables = spark.sql(f"show tables from {self._db_name}").collect()
        table_names = [table.tableName for table in tables]

        for table in tables:
            escaped_name = _resolve_names_starting_with_an_underscore(table.tableName, table_names)
            self.__setattr__(
                escaped_name,
                Table(spark, self._db_name, table.tableName, table.isTemporary),
            )

    @property
    def str(self) -> str:
        """Returns the database name."""
        return self._db_name


class Databases:
    """Loads all databases and tables in a SparkSession."""

    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        silent: bool = False,
        catalog_name: Optional[str] = None,
    ):
        spark = _get_spark_session(spark)

        if catalog_name is None:
            query = "show databases"
        else:
            query = f"show databases in {catalog_name}"

        databases = spark.sql(query).collect()
        database_names = [self._extract_db_name(database) for database in databases]
        timeout = DatabasesTimeout(silent, n=len(databases))

        for i, db_name in enumerate(database_names):
            timeout.check_for_warning(i)
            escaped_name = _resolve_names_starting_with_an_underscore(db_name, database_names)
            self.__setattr__(
                escaped_name,
                Database(spark, db_name, catalog_name),
            )

    def _extract_db_name(self, database: Row) -> str:
        """Extracts the database name from a Row.

        Old versions of Spark use ``databaseName``, newer versions use ``namespace``.
        """
        if hasattr(database, "databaseName"):  # pragma: no cover
            return database.databaseName
        if hasattr(database, "namespace"):
            return database.namespace

        raise ValueError(f"Could not find database name in {database}.")  # pragma: no cover


class Catalogs:
    """Loads all catalogs, databases and tables in a SparkSession."""

    def __init__(self, spark: Optional[SparkSession] = None, silent: bool = False):
        spark = _get_spark_session(spark)

        catalogs = spark.sql("show catalogs").collect()
        catalog_names = [catalog.catalog for catalog in catalogs]
        timeout = CatalogsTimeout(silent, n=len(catalogs))

        for i, catalog_name in enumerate(catalog_names):
            escaped_name = _resolve_names_starting_with_an_underscore(catalog_name, catalog_names)
            timeout.check_for_warning(i)
            self.__setattr__(
                escaped_name,
                Databases(spark, silent=True, catalog_name=catalog_name),
            )
