"""Defines ``LiteralTypes``, e.g. ``IntervalType.DAY``, that map their class attribute
to a ``Literal`` integer.

Can be used for example in ``DayTimeIntervalType``.
"""

from typing import Dict, Literal


class LiteralType:
    """Base class for literal types, that map their class attribute to a Literal
    integer."""

    @classmethod
    def get_dict(cls) -> Dict[str, str]:
        """Returns a dictionary mapping e.g. "IntervalType.DAY" -> "Literal[0]"."""
        dictionary = {}
        for key, value in cls.__dict__.items():
            if key.startswith("_"):
                continue

            key = f"{cls.__name__}.{key}"
            value = str(value).replace("typing.", "")

            dictionary[key] = value

        return dictionary

    @classmethod
    def get_inverse_dict(cls) -> Dict[str, str]:
        """Returns a dictionary mapping e.g. "Literal[0]" -> "IntervalType.DAY"."""
        return {v: k for k, v in cls.get_dict().items()}


class IntervalType(LiteralType):
    """Interval types for ``DayTimeIntervalType``."""

    DAY = Literal[0]
    HOUR = Literal[1]
    MINUTE = Literal[2]
    SECOND = Literal[3]
