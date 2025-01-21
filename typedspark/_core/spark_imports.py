SPARK_CONNECT = False

if SPARK_CONNECT:
    from pyspark.errors.exceptions.connect import AnalysisException  # type: ignore  # noqa: F401
    from pyspark.sql.connect.column import Column  # type: ignore  # noqa: F401
    from pyspark.sql.connect.dataframe import DataFrame  # type: ignore  # noqa: F401
else:
    from pyspark.sql import Column, DataFrame  # type: ignore  # noqa: F401
    from pyspark.sql.utils import AnalysisException  # type: ignore  # noqa: F401


# import sys

# from pyspark.sql import Column, DataFrame  # type: ignore  # noqa: F401
# from pyspark.sql.utils import AnalysisException  # type: ignore  # noqa: F401

# SPARK_CONNECT = False


# def configure(spark_connect=False):
#     global SPARK_CONNECT, AnalysisException, Column, DataFrame
#     SPARK_CONNECT = spark_connect

#     from pyspark.errors.exceptions.connect import (  # pylint: disable=redefined-outer-name
#         AnalysisException,
#     )
#     from pyspark.sql.connect.column import Column  # pylint: disable=redefined-outer-name
#     from pyspark.sql.connect.dataframe import DataFrame  # pylint: disable=redefined-outer-name

#     sys.modules[__name__].AnalysisException = AnalysisException  # type: ignore
#     sys.modules[__name__].Column = Column  # type: ignore
#     sys.modules[__name__].DataFrame = DataFrame  # type: ignore
#     hoi = True
