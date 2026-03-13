import pytest

# Skip all CIM tests if the optional cim-graph dependency is not installed.
# Install it with: pip install distopf[cim]
pytest.importorskip(
    "cimgraph",
    reason="CIM tests require the 'cim' optional dependency: pip install distopf[cim]",
)
