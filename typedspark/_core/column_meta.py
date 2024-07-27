"""Metadata for ``Column`` objects that can be accessed during runtime."""

from dataclasses import asdict, dataclass
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

    def get_metadata(self) -> Optional[Dict[str, str]]:
        """Returns the metadata of this column."""
        res = {k: v for k, v in asdict(self).items() if v is not None}
        return res if len(res) > 0 else None
