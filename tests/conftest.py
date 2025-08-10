import os
import sys

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """Fixture for creating a spark session."""
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ.pop("SPARK_REMOTE", None)
    os.environ.pop("PYSPARK_CONNECT_MODE_ENABLED", None)

    SparkSession._instantiatedSession = None  # clear any existing session
    spark = SparkSession.builder.master("local[2]").getOrCreate()
    yield spark
    spark.stop()
