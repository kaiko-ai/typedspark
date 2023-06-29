============
Contributing
============

We welcome contributions! To set up your development environment, we recommend using pyenv. You can find more on how to install ``pyenv`` and ``pyenv-virtualen`` here:

* https://github.com/pyenv/pyenv
* https://github.com/pyenv/pyenv-virtualenv

To set up the environment, run:

.. code-block:: bash

    pyenv install 3.11
    pyenv virtualenv 3.11 typedspark
    pyenv activate typedspark
    pip install -r requirements.txt
    pip install -r requirements-dev.txt

For a list of currently supported Python versions, we refer to ``.github/workflows/build.yml``.

Note that in order to run the unit tests, you will need to set up Spark on your machine.

---------------
Pre-commit hook
---------------
We use ``pre-commit`` to run a number of checks on the code before it is committed. To install the pre-commit hook, run:

.. code-block:: bash

    pre-commit install

Note that this will require you to set up Spark on your machine.

There are currently two steps from the CI/CD that we do not check using the pre-commit hook:

* bandit
* notebooks

Since they rarely fail, this shouldn't be a problem. We recommend that you test these using the CI/CD pipeline.

---------
Notebooks
---------
If you make changes that affect the documentation, please rerun the documentation notebooks in ``docs/source/``. You can do so by running the following command in the root of the repository:

.. code-block:: bash

    sh docs/run_notebooks.sh

This will run all notebooks and strip the metadata afterwards, such that the diffs in the PR remain manageable.

----------------------
Building documentation
----------------------

You can build the documentation locally by running:

.. code-block:: bash

    cd docs/; make clean; make html; cd ..

You can find the documentation in ``docs/build/html/index.html``.
