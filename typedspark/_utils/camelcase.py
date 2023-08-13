"""Utility function for converting from snake case to camel case."""


def to_camel_case(name: str) -> str:
    """Converts a string to camel case."""
    return "".join([word.capitalize() for word in name.split("_")])
