"""Removes the metadata from a notebook.

Also removes the spark warnings from cells where the sparksession is initialized.
"""

import sys

import nbformat


def clear_metadata(cell):
    """Clears the metadata of a notebook cell."""
    cell.metadata = {}


def remove_spark_warnings(cell):
    """Removes the spark warnings from a notebook cell."""
    if "outputs" in cell.keys():
        outputs = []
        for output in cell.outputs:
            if "text" in output.keys():
                if 'Setting default log level to "WARN"' in output.text:
                    continue
                if (
                    "WARN NativeCodeLoader: Unable to load native-hadoop library for your platform."
                    in output.text
                ):
                    continue
                if "WARN Utils: Service 'SparkUI' could not bind on port" in output.text:
                    continue
                if (
                    "FutureWarning: is_datetime64tz_dtype is deprecated and will be removed in a future version."  # noqa: E501
                    in output.text
                ):
                    continue
            outputs.append(output)

        cell.outputs = outputs


if __name__ == "__main__":
    FILENAME = sys.argv[1]
    nb = nbformat.read(FILENAME, as_version=4)

    for nb_cell in nb["cells"]:
        clear_metadata(nb_cell)
        remove_spark_warnings(nb_cell)

    nbformat.write(nb, FILENAME)
