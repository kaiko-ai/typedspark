"""Metadata for ``Column`` objects that can be accessed during runtime."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ColumnMeta:
    """Contains the metadata for a ``Column``. Used as:

    .. code-block:: python

        class A(Schema):
            a: Annotated[
                Column[IntegerType],
                ColumnMeta(
                    comment="This is a comment",
                )
            ]
    """

    comment: Optional[str] = None
    external_name: Optional[str] = None

    def get_metadata(self) -> Optional[Dict[str, str]]:
        """Returns the metadata of this column."""
        res = {}
        if self.comment:
            res["comment"] = self.comment
        if self.external_name:
            res["external_name"] = self.external_name

        return res if res else None
