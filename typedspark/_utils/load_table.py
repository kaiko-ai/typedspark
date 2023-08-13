"""Functions for loading `DataSet` and `Schema` in notebooks."""

import re
from typing import Dict, Literal, Optional, Tuple, Type

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType as SparkArrayType
from pyspark.sql.types import DataType
from pyspark.sql.types import DayTimeIntervalType as SparkDayTimeIntervalType
from pyspark.sql.types import DecimalType as SparkDecimalType
from pyspark.sql.types import MapType as SparkMapType
from pyspark.sql.types import StructType as SparkStructType

from typedspark._core.column import Column
from typedspark._core.dataset import DataSet
from typedspark._core.datatypes import (
    ArrayType,
    DayTimeIntervalType,
    DecimalType,
    MapType,
    StructType,
)
from typedspark._schema.schema import MetaSchema, Schema
from typedspark._utils.camelcase import to_camel_case
from typedspark._utils.register_schema_to_dataset import register_schema_to_dataset


def _replace_illegal_column_names(dataframe: DataFrame) -> DataFrame:
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


def _create_schema(structtype: SparkStructType, schema_name: Optional[str] = None) -> Type[Schema]:
    """Dynamically builds a ``Schema`` based on a ``DataFrame``'s
    ``StructType``"""
    type_annotations = {}
    attributes: Dict[str, None] = {}
    for column in structtype:
        name = column.name
        data_type = _extract_data_type(column.dataType, name)
        type_annotations[name] = Column[data_type]  # type: ignore
        attributes[name] = None

    if not schema_name:
        schema_name = "DynamicallyLoadedSchema"

    schema = MetaSchema(schema_name, tuple([Schema]), attributes)
    schema.__annotations__ = type_annotations

    return schema  # type: ignore


def _extract_data_type(dtype: DataType, name: str) -> Type[DataType]:
    """Given an instance of a ``DataType``, it extracts the corresponding
    ``DataType`` class, potentially including annotations (e.g.
    ``ArrayType[StringType]``)."""
    if isinstance(dtype, SparkArrayType):
        element_type = _extract_data_type(dtype.elementType, name)
        return ArrayType[element_type]  # type: ignore

    if isinstance(dtype, SparkMapType):
        key_type = _extract_data_type(dtype.keyType, name)
        value_type = _extract_data_type(dtype.valueType, name)
        return MapType[key_type, value_type]  # type: ignore

    if isinstance(dtype, SparkStructType):
        subschema = _create_schema(dtype, to_camel_case(name))
        return StructType[subschema]  # type: ignore

    if isinstance(dtype, SparkDayTimeIntervalType):
        start_field = dtype.startField
        end_field = dtype.endField
        return DayTimeIntervalType[Literal[start_field], Literal[end_field]]  # type: ignore

    if isinstance(dtype, SparkDecimalType):
        precision = dtype.precision
        scale = dtype.scale
        return DecimalType[Literal[precision], Literal[scale]]  # type: ignore

    return type(dtype)


def create_schema(
    dataframe: DataFrame, schema_name: Optional[str] = None
) -> Tuple[DataSet[Schema], Type[Schema]]:
    """This function inferres a ``Schema`` in a notebook based on a the provided ``DataFrame``.

    This allows for autocompletion on column names, amongst other
    things.

    .. code-block:: python

        df, Person = create_schema(df)
    """
    dataframe = _replace_illegal_column_names(dataframe)
    schema = _create_schema(dataframe.schema, schema_name)
    dataset = DataSet[schema](dataframe)  # type: ignore
    schema = register_schema_to_dataset(dataset, schema)
    return dataset, schema


def load_table(
    spark: SparkSession, table_name: str, schema_name: Optional[str] = None
) -> Tuple[DataSet[Schema], Type[Schema]]:
    """This function loads a ``DataSet``, along with its inferred ``Schema``,
    in a notebook.

    This allows for autocompletion on column names, amongst other
    things.

    .. code-block:: python

        df, Person = load_table(spark, "path.to.table")
    """
    dataframe = spark.table(table_name)
    return create_schema(dataframe, schema_name)
