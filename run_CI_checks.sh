#/bin/bash

set -e

echo "\n---- CODE STYLE CHECK ----\n"
echo "Running black\n"
poetry run black --check --diff .
echo "Running pylint\n"
poetry run pylint *.py

echo "---- UNIT TESTS ----\n"
# poetry run pytest sparkml_base_classes_test.py
