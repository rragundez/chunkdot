#/bin/bash

set -e

echo "\n---- CODE STYLE CHECK ----\n"

echo "Running ruff over the source code\n"
poetry run ruff check chunkdot

echo "Running ruff over the tests\n"
poetry run ruff check tests --ignore D103

# echo "Running pylint over the tests\n"
# poetry run pylint tests --disable C0115,C0116,W0212,R0801

echo "\n---- UNIT TESTS ----\n"

echo "\nRunning pytest\n"
poetry run pytest tests
