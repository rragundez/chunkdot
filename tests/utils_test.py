from chunkdot import utils


def test_is_package_installed():
    """Tests function that checks if a package is installed."""
    assert utils.is_package_installed("numpy")
    assert not utils.is_package_installed("some_non_existent_package")
    assert utils.is_package_installed("sklearn")
