"""Loads all databases and tables in a SparkSession."""

from datetime import datetime
from typing import Tuple, TypeVar
from warnings import warn

from pyspark.sql import Row, SparkSession

from typedspark._core.dataset import DataSet
from typedspark._schema.schema import Schema
from typedspark._utils.camelcase import to_camel_case
from typedspark._utils.load_table import load_table

T = TypeVar("T", bound=Schema)

TIMEOUT_WARNING = """
Databases() is taking longer than 10 seconds. So far, {} out of {} databases have loaded.
If this is too slow, consider loading a single database using:

from typedspark import Database

DB = Database(spark, db_name=...)
"""


class Timeout:
    """Warns the user if Databases() is taking too long."""

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
            warn(TIMEOUT_WARNING.format(i, self._n))
            self._silent = True


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


class Database:
    """Loads all tables in a database."""

    def __init__(self, spark: SparkSession, db_name: str):
        self._db_name = db_name

        tables = spark.sql(f"show tables from {db_name}").collect()
        for table in tables:
            table_name = table.tableName
            self.__setattr__(
                table_name,
                Table(spark, db_name, table_name, table.isTemporary),
            )

    @property
    def str(self) -> str:
        """Returns the database name."""
        return self._db_name


class Databases:
    """Loads all databases and tables in a SparkSession."""

    def __init__(self, spark: SparkSession, silent: bool = False):
        databases = spark.sql("show databases").collect()
        timeout = Timeout(silent, n=len(databases))

        for i, database in enumerate(databases):
            timeout.check_for_warning(i)
            db_name = self._extract_db_name(database)
            self.__setattr__(db_name, Database(spark, db_name))

    def _extract_db_name(self, database: Row) -> str:
        """Extracts the database name from a Row.

        Old versions of Spark use ``databaseName``, newer versions use ``namespace``.
        """
        if hasattr(database, "databaseName"):  # pragma: no cover
            return database.databaseName
        if hasattr(database, "namespace"):
            return database.namespace

        raise ValueError(f"Could not find database name in {database}.")  # pragma: no cover
