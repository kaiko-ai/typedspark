"""Module to output a string with the ``Schema`` definition of a given
``DataFrame``."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Type, get_args, get_origin, get_type_hints

from typedspark._core.datatypes import DayTimeIntervalType, StructType, TypedSparkDataType
from typedspark._core.literaltype import IntervalType, LiteralType
from typedspark._schema.get_schema_imports import get_schema_imports

if TYPE_CHECKING:  # pragma: no cover
    from typedspark._schema.schema import Schema


def get_schema_definition_as_string(
    schema: Type[Schema],
    include_documentation: bool,
    generate_imports: bool,
    add_subschemas: bool,
    class_name: str = "MyNewSchema",
) -> str:
    """Return the code for a given ``Schema`` as a string.

    Typically used when you load a dataset using
    ``load_dataset_from_table()`` in a notebook and you want to save the
    schema in your code base. When ``generate_imports`` is True, the
    required imports for the schema are included in the string.
    """
    imports = get_schema_imports(schema, include_documentation) if generate_imports else ""
    schema_string = _build_schema_definition_string(
        schema, include_documentation, add_subschemas, class_name
    )

    return imports + schema_string


def _build_schema_definition_string(
    schema: Type[Schema],
    include_documentation: bool,
    add_subschemas: bool,
    class_name: str = "MyNewSchema",
) -> str:
    """Return the code for a given ``Schema`` as a string."""
    lines = f"class {class_name}(Schema):\n"
    if include_documentation:
        lines += '    """Add documentation here."""\n\n'

    for k, val in get_type_hints(schema).items():
        typehint = (
            str(val)
            .replace("typedspark._core.column.", "")
            .replace("typedspark._core.datatypes.", "")
            .replace("typedspark._schema.schema.", "")
            .replace("pyspark.sql.types.", "")
            .replace("typing.", "")
        )
        typehint = _replace_literals(
            typehint, replace_literals_in=DayTimeIntervalType, replace_literals_by=IntervalType
        )
        if include_documentation:
            lines += f'    {k}: Annotated[{typehint}, ColumnMeta(comment="")]\n'
        else:
            lines += f"    {k}: {typehint}\n"

    if add_subschemas:
        lines += _add_subschemas(schema, add_subschemas, include_documentation)

    return lines


def _replace_literals(
    typehint: str,
    replace_literals_in: Type[TypedSparkDataType],
    replace_literals_by: Type[LiteralType],
) -> str:
    """Replace all Literals in a LiteralType, e.g.

    "DayTimeIntervalType[Literal[0], Literal[1]]" ->
    "DayTimeIntervalType[IntervalType.DAY, IntervalType.HOUR]"
    """
    mapping = replace_literals_by.get_inverse_dict()
    for original, replacement in mapping.items():
        typehint = _replace_literal(typehint, replace_literals_in, original, replacement)

    return typehint


def _replace_literal(
    typehint: str,
    replace_literals_in: Type[TypedSparkDataType],
    original: str,
    replacement: str,
) -> str:
    """Replaces a single Literal in a LiteralType, e.g.

    "DayTimeIntervalType[Literal[0], Literal[1]]" ->
    "DayTimeIntervalType[IntervalType.DAY, Literal[1]]"
    """
    return re.sub(
        rf"{replace_literals_in.get_name()}\[[^]]*\]",
        lambda x: x.group(0).replace(original, replacement),
        typehint,
    )


def _add_subschemas(schema: Type[Schema], add_subschemas: bool, include_documentation: bool) -> str:
    """Identifies whether any ``Column`` are of the ``StructType`` type and
    generates their schema recursively."""
    lines = ""
    for val in get_type_hints(schema).values():
        args = get_args(val)
        if not args:
            continue

        dtype = args[0]
        if get_origin(dtype) == StructType:
            lines += "\n\n"
            subschema: Type[Schema] = get_args(dtype)[0]
            lines += _build_schema_definition_string(
                subschema, include_documentation, add_subschemas, subschema.get_schema_name()
            )

    return lines
