Spark Connect support
=====================

When you connect to a remote Spark cluster via Spark Connect, PySpark transparently swaps the
classic ``DataFrame`` and ``Column`` classes for the variants under ``pyspark.sql.connect``.
``typedspark`` detects whichever class your session is using and attaches its ``DataSet`` /
``Column`` behaviour on top of it, so the same schema-aware code works against either
backend.

Spark Connect is supported when using PySpark 4.x.

Pointing your session at a Connect server
-----------------------------------------

Use ``SparkSession.builder.remote(...)`` instead of ``getOrCreate()`` on a local session:

.. code-block:: python

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.remote("sc://localhost:15002").getOrCreate()

Using typedspark on Spark Connect
---------------------------------

From there, all of the usual typedspark APIs work as you'd expect:

.. code-block:: python

    from pyspark.sql.types import LongType, StringType
    from typedspark import Column, DataSet, Schema, create_partially_filled_dataset


    class Person(Schema):
        id: Column[LongType]
        name: Column[StringType]
        age: Column[LongType]


    df: DataSet[Person] = create_partially_filled_dataset(
        spark,
        Person,
        {Person.id: [1, 2, 3], Person.name: ["John", "Jane", "Jack"]},
    )


    def birthday(df: DataSet[Person]) -> DataSet[Person]:
        return DataSet[Person](df.withColumn(Person.age.str, Person.age + 1))


    birthday(df).show()

Schema validation, ``transform_to_schema()``, ``create_empty_dataset()`` and the type-narrowed
``DataSet`` overloads (``filter``, ``unionByName``, ...) all behave the same on Spark Connect
as they do in classic PySpark — the underlying ``DataFrame`` simply remains a Connect
``DataFrame`` after each call.

Running the Connect test suite
------------------------------

The Connect-specific test in ``tests/_core/test_pyspark4_dispatch.py`` is skipped unless the
environment variable ``SPARK_CONNECT_URL`` is set. Pointing it at a running Connect server
lets you exercise typedspark end-to-end against that backend:

.. code-block:: bash

    export SPARK_CONNECT_URL="sc://localhost:15002"
    pytest tests/_core/test_pyspark4_dispatch.py
