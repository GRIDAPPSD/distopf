import pytest


def pytest_collection_modifyitems(items):
    # These tests build and solve scalar CVXPY models for parity; they are very slow.
    for item in items:
        if "tests/cvxpy_models_performance/" in item.nodeid.replace("\\", "/"):
            item.add_marker(pytest.mark.slow)
