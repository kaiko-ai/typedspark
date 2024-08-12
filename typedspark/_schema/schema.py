"""Module containing classes and functions related to TypedSpark Schemas."""

import inspect
import re
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    Union,
    _ProtocolMeta,
    get_args,
    get_type_hints,
)

from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, StructType

from typedspark._core.column import Column
from typedspark._schema.dlt_kwargs import DltKwargs
from typedspark._schema.get_schema_definition import get_schema_definition_as_string
from typedspark._schema.structfield import get_structfield


class MetaSchema(_ProtocolMeta):  # type: ignore
    """``MetaSchema`` is the metaclass of ``Schema``.

    It basically implements all functionality of ``Schema``. But since
    classes are typically considered more convenient than metaclasses,
    we provide ``Schema`` as the public interface.

    .. code-block:: python

        class A(Schema):
            a: Column[IntegerType]
            b: Column[StringType]

        DataSet[A](df)

    The class methods of ``Schema`` are described here.
    """

    _parent: Optional[Union[DataFrame, Column]] = None
    _alias: Optional[str] = None
    _current_id: Optional[int] = None
    _original_name: Optional[str] = None

    def __new__(cls, name: str, bases: Any, dct: Dict[str, Any]):
        cls._attributes = dir(cls)

        # initializes all uninitialied variables with a type annotation as None
        # this allows for auto-complete in Databricks notebooks (uninitialized variables
        # don't show up in auto-complete there).
        if "__annotations__" in dct.keys():
            extra = {attr: None for attr in dct["__annotations__"] if attr not in dct}
            dct = dict(dct, **extra)

        return super().__new__(cls, name, bases, dct)

    def __repr__(cls) -> str:
        return f"\n{str(cls)}"

    def __str__(cls) -> str:
        return cls.get_schema_definition_as_string(add_subschemas=False)

    def __getattribute__(cls, name: str) -> Any:
        """Python base function that gets attributes.

        We listen here for anyone getting a ``Column`` from the ``Schema``.
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
        if (
            name.startswith("__")
            or name == "_attributes"
            or name in cls._attributes
            or name in dir(Protocol)
        ):
            return object.__getattribute__(cls, name)

        if name in get_type_hints(cls):
            return Column(
                name,
                dtype=cls._get_dtype(name),  # type: ignore
                parent=cls._parent,
                curid=cls._current_id,
                alias=cls._alias,
            )

        raise TypeError(f"Schema {cls.get_schema_name()} does not have attribute {name}.")

    def _get_dtype(cls, name: str) -> Type[DataType]:
        """Returns the datatype of a column, e.g. Column[IntegerType] -> IntegerType."""
        column = get_type_hints(cls)[name]
        args = get_args(column)

        if not args:
            raise TypeError(
                f"Column {cls.get_schema_name()}.{name} does not have an annotated type."
            )

        dtype = args[0]
        return dtype

    def all_column_names(cls) -> List[str]:
        """Returns all column names for a given schema."""
        return list(get_type_hints(cls).keys())

    def all_column_names_except_for(cls, except_for: List[str]) -> List[str]:
        """Returns all column names for a given schema except for the columns specified
        in the ``except_for`` parameter."""
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
        add_subschemas: bool = True,
    ) -> str:
        """Return the code for the ``Schema`` as a string."""
        if schema_name is None:
            schema_name = cls.get_schema_name()
        return get_schema_definition_as_string(
            cls,  # type: ignore
            include_documentation,
            generate_imports,
            add_subschemas,
            schema_name,
        )

    def print_schema(
        cls,
        schema_name: Optional[str] = None,
        include_documentation: bool = False,
        generate_imports: bool = True,
        add_subschemas: bool = False,
    ):  # pragma: no cover
        """Print the code for the ``Schema``."""
        print(
            cls.get_schema_definition_as_string(
                schema_name=schema_name,
                include_documentation=include_documentation,
                generate_imports=generate_imports,
                add_subschemas=add_subschemas,
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
        """Creates a representation of the ``Schema`` to be used by Delta Live Tables.

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

    def get_metadata(cls) -> dict[str, dict[str, Any]]:
        """Returns the metadata of each of the columns in the schema."""
        return {field.name: field.metadata for field in cls.get_structtype().fields}


class Schema(Protocol, metaclass=MetaSchema):
    # pylint: disable=empty-docstring
    # Since docstrings are inherrited, and since we use docstrings to
    # annotate tables (see MetaSchema.get_dlt_kwargs()), we have chosen
    # to add an empty docstring to the Schema class (otherwise the Schema
    # docstring would be added to any schema without a docstring).
    """"""
