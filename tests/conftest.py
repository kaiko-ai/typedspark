import os
import sys

import pytest
from pyspark.sql import SparkSession


def pytest_addoption(parser):
    parser.addoption(
        "--spark-connect",
        action="store_true",
        default=False,
        help="Run the unit tests using a spark-connect session.",
    )


@pytest.fixture(scope="session")
def spark(pytestconfig: pytest.Config):
    """Fixture for creating a spark session."""
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark_connect = pytestconfig.getoption("--spark-connect")
    if spark_connect:
        # from typedspark import configure

        # configure(spark_connect=True)

        spark = (
            SparkSession.Builder()
            .config("spark.jars.packages", "org.apache.spark:spark-connect_2.12:3.5.3")
            .remote("local")
            .getOrCreate()
        )
    else:
        spark = SparkSession.Builder().getOrCreate()

    yield spark
    spark.stop()
