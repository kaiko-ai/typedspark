============
Contributing
============

We welcome contributions! To set up your development environment, we recommend using pyenv. You can find more on how to install ``pyenv`` and ``pyenv-virtualen`` here:

* https://github.com/pyenv/pyenv
* https://github.com/pyenv/pyenv-virtualenv

To set up the environment, run:

.. code-block:: bash

    pyenv install 3.10
    pyenv virtualenv 3.10 typedspark
    pyenv activate typedspark
    pip install -r requirements.txt
    pip install -r requirements-dev.txt

For a list of currently supported Python versions, as well as the various CI/CD steps, we refer to ``.github/workflows/build.yml``.

Note that in order to run the unit tests, you will need to set up Spark on your machine.
