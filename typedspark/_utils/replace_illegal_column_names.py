import re
from typing import Dict

from pyspark.sql import DataFrame


def replace_illegal_column_names(dataframe: DataFrame) -> DataFrame:
    """Replaces illegal column names with a legal version."""
    mapping = _create_mapping(dataframe)

    for column, column_renamed in mapping.items():
        if column != column_renamed:
            dataframe = dataframe.withColumnRenamed(column, column_renamed)

    return dataframe


def _create_mapping(dataframe: DataFrame) -> Dict[str, str]:
    """Checks if there are duplicate columns after replacing illegal characters."""
    mapping = {column: _replace_illegal_characters(column) for column in dataframe.columns}
    renamed_columns = list(mapping.values())
    duplicates = {
        column: column_renamed
        for column, column_renamed in mapping.items()
        if renamed_columns.count(column_renamed) > 1
    }

    if len(duplicates) > 0:
        raise ValueError(
            "You're trying to dynamically generate a Schema from a DataFrame. "
            + "However, typedspark has detected that the DataFrame contains duplicate columns "
            + "after replacing illegal characters (e.g. whitespaces, dots, etc.).\n"
            + "The folowing columns have lead to duplicates:\n"
            + f"{duplicates}\n\n"
            + "Please rename these columns in your DataFrame."
        )

    return mapping


def _replace_illegal_characters(column_name: str) -> str:
    """Replaces illegal characters in a column name with an underscore."""
    return re.sub("[^A-Za-z0-9]", "_", column_name)
