"""Module containing classes and functions related to TypedSpark Schemas."""
import inspect
import re
from typing import Any, Dict, List, Optional, Union, get_type_hints

from pyspark.sql import DataFrame
from pyspark.sql.types import StructType

from typedspark._core.column import Column
from typedspark._schema.dlt_kwargs import DltKwargs
from typedspark._schema.get_schema_definition import get_schema_definition_as_string
from typedspark._schema.structfield import get_structfield


class MetaSchema(type):
    """`MetaSchema` is the metaclass of `Schema`.

    It basically implements all functionality of `Schema`. But since
    people are more comfortable with classes (rather than metaclasses),
    we provide `Schema` as the public interface. Should be used as:

    .. code-block:: python
        class A(Schema):
            a: Column[IntegerType]
            b: Column[StringType]

        DataSet[A](df)
    """

    _linked_dataframe: Optional[DataFrame] = None
    _current_id: Optional[int] = None
    _original_name: Optional[str] = None

    def __new__(cls, name: str, bases: Any, dct: Dict[str, Any]):
        # initializes all uninitialied variables with a type annotation as None
        # this allows for auto-complete in Databricks notebooks (uninitialized variables
        # don't show up in auto-complete there).
        if "__annotations__" in dct.keys():
            extra = {attr: None for attr in dct["__annotations__"] if attr not in dct}
            dct = dict(dct, **extra)

        return type.__new__(cls, name, bases, dct)

    def __repr__(cls) -> str:
        return f"\n{str(cls)}"

    def __str__(cls) -> str:
        return cls.get_schema_definition_as_string()

    def __getattribute__(cls, name: str) -> Any:
        """Python base function that gets attributes.

        We listen here for anyone getting `Column`s from the `Schema`.
        Even though they're not explicitely instantiated, we can instantiate
        them here whenever someone attempts to get them. This allows us to do the following:

        .. code-block:: python
            class A(Schema):
                a: Column[IntegerType]

            (
                df.withColumn(A.a.str, lit(1))
                .select(A.a)
            )
        """
        if name.startswith("__") or name in [
            "all_column_names",
            "all_column_names_except_for",
            "get_docstring",
            "get_dlt_kwargs",
            "get_primary_key_names",
            "get_schema_definition_as_string",
            "get_snake_case",
            "get_structtype",
            "get_schema_name",
            "print_schema",
            "_current_id",
            "_linked_dataframe",
            "_original_name",
        ]:
            return object.__getattribute__(cls, name)

        if name in get_type_hints(cls).keys():
            return Column(name, cls._linked_dataframe, cls._current_id)

        raise TypeError(f"Schema {cls.get_schema_name()} does not have attribute {name}.")

    def all_column_names(cls) -> List[str]:
        """Returns all column names for a given schema."""
        return list(get_type_hints(cls).keys())

    def all_column_names_except_for(cls, except_for: List[str]) -> List[str]:
        """Returns all column names for a given schema except for the columns
        specified in the `except_for` parameter."""
        return list(name for name in get_type_hints(cls).keys() if name not in except_for)

    def get_snake_case(cls) -> str:
        """Return the class name transformed into snakecase."""
        word = cls.get_schema_name()
        word = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", word)
        word = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", word)
        word = word.replace("-", "_")
        return word.lower()

    def get_schema_definition_as_string(
        cls,
        schema_name: Optional[str] = None,
        include_documentation: bool = False,
        generate_imports: bool = True,
    ) -> str:
        """Return the code for the `Schema` as a string."""
        if schema_name is None:
            schema_name = cls.get_schema_name()
        return get_schema_definition_as_string(
            cls,  # type: ignore
            include_documentation,
            generate_imports,
            schema_name,
        )

    def print_schema(
        cls,
        schema_name: Optional[str] = None,
        include_documentation: bool = False,
        generate_imports: bool = True,
    ):  # pragma: no cover
        """Print the code for the `Schema`, including the template for
        documentation (i.e. docstring and `ColumMeta(comment="")`)."""
        print(
            cls.get_schema_definition_as_string(
                schema_name=schema_name,
                include_documentation=include_documentation,
                generate_imports=generate_imports,
            )
        )

    def get_docstring(cls) -> Union[str, None]:
        """Returns the docstring of the schema."""
        return inspect.getdoc(cls)

    def get_structtype(cls) -> StructType:
        """Creates the spark StructType for the schema."""
        return StructType(
            [
                get_structfield(name, column)
                for name, column in get_type_hints(cls, include_extras=True).items()
            ]
        )

    def get_dlt_kwargs(cls, name: Optional[str] = None) -> DltKwargs:
        """Creates a representation of the `Schema` to be used by Delta Live
        Tables.

        .. code-block:: python
            @dlt.table(**DimPatient.get_dlt_kwargs())
            def table_definition() -> DataSet[DimPatient]:
                <your table definition here>
        """
        return {
            "name": name if name else cls.get_snake_case(),
            "comment": cls.get_docstring(),
            "schema": cls.get_structtype(),
        }

    def get_schema_name(cls):
        """Returns the name with which the schema was initialized."""
        return cls._original_name if cls._original_name else cls.__name__


class Schema(metaclass=MetaSchema):
    # pylint: disable=missing-class-docstring
    # Since docstrings are inherrited, and since we use docstrings to
    # annotate tables (see MetaSchema.get_dlt_kwargs()), we have chosen
    # to not add a docstring to the Schema class (otherwise the Schema
    # docstring would be added to any schema without a docstring).
    pass
