#/bin/bash

set -e

update_version_type=$1

# update version
poetry version $update_version_type
version=$(poetry version --short)
line_number=$(grep -n "__version__" chunkdot/__init__.py | cut -d: -f1)
sed -i '' "${line_number}s/.*/__version__ = \"${version}\"/" chunkdot/__init__.py

# check versions match in pyproject.toml and chunkot/__init__.py
poetry run pytest tests/package_version_test.py

# add, commit and tag changes of the versions
git add chunkdot/__init__.py pyproject.toml
git commit -m "Release ${version}"
git tag -a $version -m "My release ${version}"

# push changes
git push origin main
git push origin --tags

# release
poetry publish --build -v
