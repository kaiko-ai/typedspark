"""Module responsible for generating StructFields from Columns in a Schema."""

from __future__ import annotations

from typing import Annotated, Type, TypeVar, Union, get_args, get_origin

from pyspark.sql.types import DataType, StructField

from typedspark._core.column import Column
from typedspark._core.column_meta import ColumnMeta
from typedspark._core.datatypes import materialize_dtype

_DataType = TypeVar("_DataType", bound=DataType)  # pylint: disable=invalid-name


def get_structfield(
    name: str,
    column: Union[Type[Column[_DataType]], Annotated[Type[Column[_DataType]], ColumnMeta]],
) -> StructField:
    """Generates a ``StructField`` for a given ``Column`` in a ``Schema``."""
    meta = get_structfield_meta(column)

    return StructField(
        name=name,
        dataType=_get_structfield_dtype(column, name),
        nullable=True,
        metadata=meta.get_metadata(),
    )


def get_structfield_meta(
    column: Union[Type[Column[_DataType]], Annotated[Type[Column[_DataType]], ColumnMeta]],
) -> ColumnMeta:
    """Get the spark column metadata from the ``ColumnMeta`` data, when available."""
    return next((x for x in get_args(column) if isinstance(x, ColumnMeta)), ColumnMeta())


def _get_structfield_dtype(
    column: Union[Type[Column[_DataType]], Annotated[Type[Column[_DataType]], ColumnMeta]],
    colname: str,
) -> DataType:
    """Get the spark ``DataType`` from the ``Column`` type annotation."""
    origin = get_origin(column)
    if origin not in [Annotated, Column]:
        raise TypeError(f"Column {colname} needs to be of type Column or Annotated.")

    if origin == Annotated:
        column = _get_column_from_annotation(column, colname)

    args = get_args(column)
    dtype = materialize_dtype(args[0], colname)
    return dtype


def _get_column_from_annotation(
    column: Annotated[Type[Column[_DataType]], ColumnMeta],
    colname: str,
) -> Type[Column[_DataType]]:
    """Takes an ``Annotation[Column[...], ...]`` and returns the ``Column[...]``."""
    column = get_args(column)[0]
    if get_origin(column) != Column:
        raise TypeError(f"Column {colname} needs to have a Column[] within Annotated[].")

    return column
