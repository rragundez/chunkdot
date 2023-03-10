#/bin/bash

set -e

echo "\n---- CODE STYLE CHECK ----\n"
echo "Running black\n"
poetry run black --check --diff .
echo "\nRunning pylint\n"
poetry run pylint *.py

echo "\n---- UNIT TESTS ----\n"
poetry run pytest cosine_similarity_top_k_test.py
