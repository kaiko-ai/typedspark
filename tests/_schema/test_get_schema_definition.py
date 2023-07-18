from typedspark._core.datatypes import DayTimeIntervalType
from typedspark._core.literaltype import IntervalType
from typedspark._schema.get_schema_definition import _replace_literal, _replace_literals


def test_replace_literal():
    result = _replace_literal(
        "DayTimeIntervalType[Literal[0], Literal[1]]",
        replace_literals_in=DayTimeIntervalType,
        original="Literal[0]",
        replacement="IntervalType.DAY",
    )
    expected = "DayTimeIntervalType[IntervalType.DAY, Literal[1]]"

    assert result == expected


def test_replace_literals():
    result = _replace_literals(
        "DayTimeIntervalType[Literal[0], Literal[1]]",
        replace_literals_in=DayTimeIntervalType,
        replace_literals_by=IntervalType,
    )
    expected = "DayTimeIntervalType[IntervalType.DAY, IntervalType.HOUR]"

    assert result == expected
