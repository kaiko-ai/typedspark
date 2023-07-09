"""Functions for loading `DataSet` and `Schema` in notebooks."""

import re
from typing import Dict, Literal, Optional, Tuple, Type

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType as SparkArrayType
from pyspark.sql.types import DataType
from pyspark.sql.types import DecimalType as SparkDecimalType
from pyspark.sql.types import MapType as SparkMapType
from pyspark.sql.types import StructType as SparkStructType

from typedspark._core.column import Column
from typedspark._core.dataset import DataSet
from typedspark._core.datatypes import ArrayType, DecimalType, MapType, StructType
from typedspark._schema.schema import MetaSchema, Schema
from typedspark._utils.register_schema_to_dataset import register_schema_to_dataset


def _replace_illegal_column_names(dataframe: DataFrame) -> DataFrame:
    """Replaces illegal column names with a legal version."""
    for column in dataframe.columns:
        dataframe = dataframe.withColumnRenamed(column, _replace_illegal_characters(column))
    return dataframe


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
        subschema = _create_schema(dtype, _to_camel_case(name))
        return StructType[subschema]  # type: ignore

    if isinstance(dtype, SparkDecimalType):
        precision = dtype.precision
        scale = dtype.scale
        return DecimalType[Literal[precision], Literal[scale]]  # type: ignore

    return type(dtype)


def _to_camel_case(name: str) -> str:
    """Converts a string to camel case."""
    return "".join([word.capitalize() for word in name.split("_")])


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
