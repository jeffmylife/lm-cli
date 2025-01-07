from setuptools import setup, find_packages

setup(
    name="llm-cli",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "litellm>=1.30.3",
        "rich>=13.7.0",
        "typer>=0.9.0",
    ],
    entry_points={
        "console_scripts": [
            "lm=llm_cli.cli:app",
        ],
    },
    author="Jeffrey Lemoine",
    author_email="jeffmlife@gmail.com",
    description="A CLI tool for interacting with various LLM models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jeffmylife/llm-cli",
    python_requires=">=3.10",
)
