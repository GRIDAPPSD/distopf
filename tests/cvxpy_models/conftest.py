import pytest


def pytest_collection_modifyitems(items):
    # CVXPY model tests are very slow (and can crash on Windows); run only with `-m slow`.
    for item in items:
        if "tests/cvxpy_models/" in item.nodeid.replace("\\", "/"):
            item.add_marker(pytest.mark.slow)
