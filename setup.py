from setuptools import find_packages, setup


def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


def get_long_description():
    with open("README.md", encoding="utf-8") as f:
        return f.read()


setup(
    name="typedspark",
    url="https://github.com/kaiko-ai/typedspark",
    license="Apache-2.0",
    author="Nanne Aben",
    author_email="nanne@kaiko.ai",
    description="Column-wise type annotations for pyspark DataFrames",
    keywords="pyspark spark typing type checking annotations",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["typedspark", "typedspark.*"]),
    install_requires=get_requirements(),
    python_requires=">=3.9.0",
    classifiers=["Programming Language :: Python", "Typing :: Typed"],
    setuptools_git_versioning={"enabled": True},
    setup_requires=["setuptools-git-versioning>=2.0,<3"],
    package_data={"typedspark": ["py.typed"]},
    extras_require={
        "pyspark": ["pyspark"],
    },
)
