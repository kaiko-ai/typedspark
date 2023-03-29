"""A representation of the ``Schema`` to be used by Delta Live Tables."""

from typing import Optional, TypedDict

from pyspark.sql.types import StructType


class DltKwargs(TypedDict):
    """A representation of the ``Schema`` to be used by Delta Live Tables.

    .. code-block:: python

        @dlt.table(**Person.get_dlt_kwargs())
        def table_definition() -> DataSet[Person]:
            <your table definition here>
    """

    name: str
    comment: Optional[str]
    schema: StructType
