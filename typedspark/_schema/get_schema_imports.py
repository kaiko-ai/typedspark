"""Builds an import statement for everything imported by a given ``Schema``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Type, get_args, get_origin, get_type_hints

from pyspark.sql.types import DataType

from typedspark._core.datatypes import (
    ArrayType,
    DayTimeIntervalType,
    DecimalType,
    MapType,
    StructType,
    TypedSparkDataType,
)

if TYPE_CHECKING:  # pragma: no cover
    from typedspark._schema.schema import Schema


def get_schema_imports(schema: Type[Schema], include_documentation: bool) -> str:
    """Builds an import statement for everything imported by the ``Schema``."""
    dtypes = _get_imported_dtypes(schema)
    return _build_import_string(dtypes, include_documentation)


def _get_imported_dtypes(schema: Type[Schema]) -> set[Type[DataType]]:
    """Returns a set of DataTypes that are imported by the given schema."""
    encountered_datatypes: set[Type[DataType]] = set()
    for column in get_type_hints(schema).values():
        args = get_args(column)
        if not args:
            continue

        dtype = args[0]
        encountered_datatypes |= _process_datatype(dtype)

    return encountered_datatypes


def _process_datatype(dtype: Type[DataType]) -> set[Type[DataType]]:
    """Returns a set of DataTypes that are imported for a given DataType.

    Handles nested DataTypes recursively.
    """
    encountered_datatypes: set[Type[DataType]] = set()

    origin: Optional[Type[DataType]] = get_origin(dtype)
    if origin:
        encountered_datatypes.add(origin)
    else:
        encountered_datatypes.add(dtype)

    if origin == MapType:
        key, value = get_args(dtype)
        encountered_datatypes |= _process_datatype(key)
        encountered_datatypes |= _process_datatype(value)

    if origin == ArrayType:
        element = get_args(dtype)[0]
        encountered_datatypes |= _process_datatype(element)

    if get_origin(dtype) == StructType:
        subschema = get_args(dtype)[0]
        encountered_datatypes |= _get_imported_dtypes(subschema)

    return encountered_datatypes


def _build_import_string(
    encountered_datatypes: set[Type[DataType]], include_documentation: bool
) -> str:
    """Returns a multiline string with the imports required for the given
    encountered_datatypes.

    Import sorting is applied.

    If the schema uses IntegerType, BooleanType, StringType, this functions result would be

    .. code-block:: python

        from pyspark.sql.types import BooleanType, IntegerType, StringType

        from typedspark import Column, Schema
    """
    return (
        _typing_imports(encountered_datatypes, include_documentation)
        + _pyspark_imports(encountered_datatypes)
        + _typedspark_imports(encountered_datatypes, include_documentation)
    )


def _typing_imports(encountered_datatypes: set[Type[DataType]], include_documentation: bool) -> str:
    """Returns the import statement for the typing library."""
    imports = []

    if any([dtype == DecimalType for dtype in encountered_datatypes]):
        imports += ["Literal"]

    if include_documentation:
        imports += ["Annotated"]

    if len(imports) > 0:
        imports = sorted(imports)
        imports_string = ", ".join(imports)  # type: ignore
        return f"from typing import {imports_string}\n\n"

    return ""


def _pyspark_imports(encountered_datatypes: set[Type[DataType]]) -> str:
    """Returns the import statement for the pyspark library."""
    dtypes = sorted(
        [
            dtype.__name__
            for dtype in encountered_datatypes
            if not issubclass(dtype, TypedSparkDataType)
        ]
    )

    if len(dtypes) > 0:
        dtypes_string = ", ".join(dtypes)
        return f"from pyspark.sql.types import {dtypes_string}\n\n"

    return ""


def _typedspark_imports(
    encountered_datatypes: set[Type[DataType]], include_documentation: bool
) -> str:
    """Returns the import statement for the typedspark library."""
    dtypes = [
        dtype.__name__ for dtype in encountered_datatypes if issubclass(dtype, TypedSparkDataType)
    ] + ["Column", "Schema"]

    if any([dtype == DayTimeIntervalType for dtype in encountered_datatypes]):
        dtypes += ["IntervalType"]

    if include_documentation:
        dtypes.append("ColumnMeta")

    dtypes = sorted(dtypes)

    dtypes_string = ", ".join(dtypes)
    return f"from typedspark import {dtypes_string}\n\n\n"
