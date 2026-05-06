"""Execute notebooks in a single shared Jupyter kernel.

Used by CI to verify the example notebooks. Reusing one kernel means the JVM and Spark
session start once (on the first ``SparkSession.builder.getOrCreate()`` call) and are
shared across notebooks. ``%reset -f`` between notebooks clears the IPython namespace
without tearing down the JVM-side Spark session, so each notebook still has to create
its own imports and ``spark`` handle.
"""

from __future__ import annotations

import sys

import nbformat
from jupyter_client.manager import KernelManager
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def main(paths: list[str]) -> int:
    if not paths:
        print("usage: run_notebooks.py <notebook> [<notebook> ...]", file=sys.stderr)
        return 2

    km = KernelManager()
    km.start_kernel()
    failures: list[str] = []
    try:
        for path in paths:
            nb = nbformat.read(path, as_version=4)
            nb.cells.insert(0, nbformat.v4.new_code_cell("%reset -f"))
            try:
                NotebookClient(nb, km=km).execute()
            except CellExecutionError as exc:
                failures.append(path)
                print(f"FAIL: {path}\n{exc}", file=sys.stderr)
            else:
                print(f"OK:   {path}")
    finally:
        km.shutdown_kernel()

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
