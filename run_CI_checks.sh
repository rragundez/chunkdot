#/bin/bash

set -e

echo "\n---- CODE STYLE CHECK ----\n"

echo "Running black\n"
poetry run black --check --diff .

echo "\nRunning pylint over chunkdot source code\n"
poetry run pylint chunkdot

echo "Running pylint over the tests\n"
poetry run pylint tests --disable C0115,C0116,W0212,R0801

echo "\n---- UNIT TESTS ----\n"

echo "\nRunning pytest\n"
poetry run pytest tests
