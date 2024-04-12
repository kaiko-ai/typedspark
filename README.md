# Typedspark: column-wise type annotations for pyspark DataFrames

We love Spark! But in production code we're wary when we see:

```python
from pyspark.sql import DataFrame

def foo(df: DataFrame) -> DataFrame:
    # do stuff
    return df
```

Because… How do we know which columns are supposed to be in ``df``?

Using ``typedspark``, we can be more explicit about what these data should look like.

```python
from typedspark import Column, DataSet, Schema
from pyspark.sql.types import LongType, StringType

class Person(Schema):
    id: Column[LongType]
    name: Column[StringType]
    age: Column[LongType]

def foo(df: DataSet[Person]) -> DataSet[Person]:
    # do stuff
    return df
```
The advantages include:

* Improved readability of the code
* Typechecking, both during runtime and linting
* Auto-complete of column names
* Easy refactoring of column names
* Easier unit testing through the generation of empty ``DataSets`` based on their schemas
* Improved documentation of tables

## Documentation
Please see our documentation on [readthedocs](https://typedspark.readthedocs.io/en/latest/index.html).

## Installation

You can install ``typedspark`` from [pypi](https://pypi.org/project/typedspark/) by running:

```bash
pip install typedspark
```
By default, ``typedspark`` does not list ``pyspark`` as a dependency, since many platforms (e.g. Databricks) come with ``pyspark`` preinstalled.  If you want to install ``typedspark`` with ``pyspark``, you can run:

```bash
pip install "typedspark[pyspark]"
```

## Demo videos

### IDE demo

https://github.com/kaiko-ai/typedspark/assets/47976799/e6f7fa9c-6d14-4f68-baba-fe3c22f75b67

You can find the corresponding code [here](docs/videos/ide.ipynb).

### Jupyter / Databricks notebooks demo

https://github.com/kaiko-ai/typedspark/assets/47976799/39e157c3-6db0-436a-9e72-44b2062df808

You can find the corresponding code [here](docs/videos/notebook.ipynb).

## FAQ

**I found a bug! What should I do?**</br>
Great! Please make an issue and we'll look into it.

**I have a great idea to improve typedspark! How can we make this work?**</br>
Awesome, please make an issue and let us know!
