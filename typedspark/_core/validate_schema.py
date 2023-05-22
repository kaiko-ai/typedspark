"""Module containing functions that are related to validating schema's at runtime."""
from typing import Dict, Set

from pyspark.sql.types import ArrayType, DataType, MapType, StructField, StructType


def validate_schema(
    structtype_expected: StructType, structtype_observed: StructType, schema_name: str
) -> None:
    """Checks whether the expected and the observed StructType match."""
    expected = unpack_schema(structtype_expected)
    observed = unpack_schema(structtype_observed)

    check_names(set(expected.keys()), set(observed.keys()), schema_name)
    check_dtypes(expected, observed, schema_name)


def unpack_schema(schema: StructType) -> Dict[str, StructField]:
    """Converts the observed schema to a dictionary mapping column name to StructField.

    We ignore columns that start with ``__``.
    """
    res = {}
    for field in schema.fields:
        if field.name.startswith("__"):
            continue
        field.nullable = True
        field.metadata = {}
        res[field.name] = field

    return res


def check_names(names_expected: Set[str], names_observed: Set[str], schema_name: str) -> None:
    """Checks whether the observed and expected list of column names overlap.

    Is order insensitive.
    """
    diff = names_observed - names_expected
    if diff:
        raise TypeError(
            f"Data contains the following columns not present in schema {schema_name}: {diff}"
        )

    diff = names_expected - names_observed
    if diff:
        raise TypeError(
            f"Schema {schema_name} contains the following columns not present in data: {diff}"
        )


def check_dtypes(
    schema_expected: Dict[str, StructField],
    schema_observed: Dict[str, StructField],
    schema_name: str,
) -> None:
    """Checks for each column whether the observed and expected data type match.

    Is order insensitive.
    """
    for name, structfield_expected in schema_expected.items():
        structfield_observed = schema_observed[name]
        check_dtype(
            name,
            structfield_expected.dataType,
            structfield_observed.dataType,
            schema_name,
        )


def check_dtype(
    colname: str, dtype_expected: DataType, dtype_observed: DataType, schema_name: str
) -> None:
    """Checks whether the observed and expected data type match."""
    if dtype_expected == dtype_observed:
        return None

    if isinstance(dtype_expected, ArrayType) and isinstance(dtype_observed, ArrayType):
        return check_dtype(
            f"{colname}.element_type",
            dtype_expected.elementType,
            dtype_observed.elementType,
            schema_name,
        )

    if isinstance(dtype_expected, MapType) and isinstance(dtype_observed, MapType):
        check_dtype(
            f"{colname}.key",
            dtype_expected.keyType,
            dtype_observed.keyType,
            schema_name,
        )
        return check_dtype(
            f"{colname}.value",
            dtype_expected.valueType,
            dtype_observed.valueType,
            schema_name,
        )

    if isinstance(dtype_expected, StructType) and isinstance(dtype_observed, StructType):
        return validate_schema(dtype_expected, dtype_observed, f"{schema_name}.{colname}")

    raise TypeError(
        f"Column {colname} is of type {dtype_observed}, but {schema_name}.{colname} "
        + f"suggests {dtype_expected}."
    )
