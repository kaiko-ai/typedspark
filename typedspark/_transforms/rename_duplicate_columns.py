"""Module that handles duplicate columns in the ``DataFrame``, that are also in the
schema (and hence in the resulting ``DataSet[Schema]``), but which are not handled by
the transformations dictionary."""

from typing import Dict, Final, Type
from uuid import uuid4

from pyspark.sql import Column as SparkColumn

from typedspark._schema.schema import Schema

ERROR_MSG: Final[
    str
] = """Columns {columns} are ambiguous.
Please specify the transformations for these columns explicitly, for example:

schema_a = register_schema_to_dataset(df_a, A)
schema_b = register_schema_to_dataset(df_b, B)

transform_to_schema(
    df_a.join(
        df_b,
        schema_a.id == schema_b.id
    ),
    C,
    {{
        C.id: schema_a.id,
    }}
)
"""


class RenameDuplicateColumns:
    """Class that handles duplicate columns in the DataFrame, that are also in the
    schema (and hence in the resulting ``DataSet[Schema]``), but which are not handled
    by the transformations dictionary.

    This class renames these duplicate columns to temporary names, such that we avoid
    ambiguous columns in the resulting ``DataSet[Schema]``.
    """

    def __init__(
        self,
        transformations: Dict[str, SparkColumn],
        schema: Type[Schema],
        dataframe_columns: list[str],
    ):
        self._temporary_key_mapping = self._create_temporary_key_mapping(
            transformations, dataframe_columns, schema
        )
        self._transformations = self._rename_keys_to_temporary_keys(transformations)

    def _create_temporary_key_mapping(
        self,
        transformations: Dict[str, SparkColumn],
        dataframe_columns: list[str],
        schema: Type[Schema],
    ) -> Dict[str, str]:
        """Creates a mapping for duplicate columns in the ``DataFrame`` to temporary
        names, such that we avoid ambiguous columns in the resulting
        ``DataSet[Schema]``."""
        duplicate_columns_in_dataframe = self._duplicates(dataframe_columns)
        schema_columns = set(schema.all_column_names())
        transformation_keys = set(transformations.keys())

        self._verify_that_all_duplicate_columns_will_be_handled(
            duplicate_columns_in_dataframe, schema_columns, transformation_keys
        )

        problematic_keys = duplicate_columns_in_dataframe & transformation_keys

        res = {}
        for key in problematic_keys:
            res[key] = self._find_temporary_name(key, dataframe_columns)

        return {k: f"my_temporary_typedspark_{k}" for k in problematic_keys}

    def _duplicates(self, lst: list) -> set:
        """Returns a set of the duplicates in the provided list."""
        return {x for x in lst if lst.count(x) > 1}

    def _verify_that_all_duplicate_columns_will_be_handled(
        self,
        duplicate_columns_in_dataframe: set[str],
        schema_columns: set[str],
        transformation_keys: set[str],
    ):
        """Raises an exception if there are duplicate columns in the ``DataFrame``, that
        are also in the schema (and hence in the resulting ``DataSet[Schema]``), but
        which are not handled by the ``transformation``s`` dictionary."""
        unhandled_columns = (duplicate_columns_in_dataframe & schema_columns) - transformation_keys
        if unhandled_columns:
            raise ValueError(ERROR_MSG.format(columns=unhandled_columns))

    def _find_temporary_name(self, colname: str, dataframe_columns: list[str]) -> str:
        """Appends a uuid to the column name to make sure the temporary name doesn't
        collide with any other column names."""
        name = colname
        num = 0
        while name in dataframe_columns:
            name = f"{colname}_with_temporary_uuid_{uuid4()}"
            num += 1
            if num > 100:
                raise Exception("Failed to find a temporary name.")  # pragma: no cover

        return name

    def _rename_keys_to_temporary_keys(
        self, transformations: Dict[str, SparkColumn]
    ) -> Dict[str, SparkColumn]:
        """Renames the keys in the transformations dictionary to temporary keys."""
        return {self._temporary_key_mapping.get(k, k): v for k, v in transformations.items()}

    @property
    def transformations(self) -> Dict[str, SparkColumn]:
        """Returns the transformations dictionary."""
        return self._transformations

    @property
    def temporary_key_mapping(self) -> Dict[str, str]:
        """Returns the temporary key mapping."""
        return self._temporary_key_mapping
