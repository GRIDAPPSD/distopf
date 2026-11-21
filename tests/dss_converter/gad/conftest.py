"""Fixtures for the GAD converter tests."""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def gad_devices_dss() -> Path:
    """Small OpenDSS model with every GAD device type (see the header of the file)."""
    path = DATA_DIR / "gad_devices.dss"
    assert path.exists(), f"missing test model {path}"
    return path
