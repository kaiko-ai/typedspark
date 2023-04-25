===============================================================
Typedspark: column-wise type annotations for pyspark DataFrames
===============================================================

We love Spark! But in production code we're wary when we see:

.. code-block:: python

    from pyspark.sql import DataFrame

    def foo(df: DataFrame) -> DataFrame:
        # do stuff
        return df

Because… How do we know which columns are supposed to be in ``df``?

Using ``typedspark``, we can be more explicit about what these data should look like.

.. code-block:: python

    from typedspark import Column, DataSet, Schema
    from pyspark.sql.types import LongType, StringType

    class Person(Schema):
        id: Column[LongType]
        name: Column[StringType]
        age: Column[LongType]

    def foo(df: DataSet[Person]) -> DataSet[Person]:
        # do stuff
        return df

The advantages include:

* Improved readibility of the code
* Typechecking, both during runtime and linting
* Auto-complete of column names
* Easy refactoring of column names
* Easier unit testing through the generation of empty ``DataSets`` based on their schemas
* Improved documentation of tables

Installation
============

You can install ``typedspark`` from `pypi <https://pypi.org/project/typedspark/>`_ by running:

.. code-block:: bash

    pip install typedspark

By default, ``typedspark`` does not list ``pyspark`` as a dependency, since many platforms (e.g. Databricks) come with ``pyspark`` preinstalled.  If you want to install ``typedspark`` with ``pyspark``, you can run:

.. code-block:: bash

    pip install "typedspark[pyspark]"


Documentation
=================
Please see our documentation on `readthedocs <https://typedspark.readthedocs.io/en/latest/index.html>`_.

FAQ
===

| **I found a bug! What should I do?**
| Great! Please make an issue and we'll look into it.
|
| **I have a great idea to improve typedspark! How can we make this work?**
| Awesome, please make an issue and let us know!
