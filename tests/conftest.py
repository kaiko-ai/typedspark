import os
import sys

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """Fixture for creating a spark session."""
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = SparkSession.Builder().getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def sparkConnect():
    """Fixture for creating a spark session."""

    spark = (
        SparkSession.Builder()
        .config("spark.jars.packages", "org.apache.spark:spark-connect_2.12:3.5.3")
        .remote('local')
        .getOrCreate()
    )
    yield spark
    spark.stop()
