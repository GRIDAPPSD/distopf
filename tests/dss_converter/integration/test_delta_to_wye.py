"""delta_to_wye rewrites 3-phase transformer windings to wye in the loaded circuit."""

from pathlib import Path

from distopf.dss_importer.dss_to_csv_converter import DSSToCSVConverter

DSS_FILE = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "distopf"
    / "cases"
    / "dss"
    / "4Bus-DY-Bal"
    / "4Bus-DY-Bal.DSS"
)


def _winding_connections(converter: DSSToCSVConverter) -> list[int]:
    """Return DSS winding connection codes (0 = wye, 1 = delta) of transformer t1."""
    converter.dss.Transformers.Name("t1")
    connections = []
    for winding in (1, 2):
        converter.dss.Transformers.Wdg(winding)
        connections.append(converter.dss.Transformers.IsDelta())
    return [int(is_delta) for is_delta in connections]


def test_default_keeps_the_delta_winding():
    converter = DSSToCSVConverter(DSS_FILE)
    assert _winding_connections(converter) == [1, 0]


def test_delta_to_wye_makes_every_winding_wye():
    converter = DSSToCSVConverter(DSS_FILE, delta_to_wye=True)
    assert _winding_connections(converter) == [0, 0]
    assert len(converter.branch_data) == len(DSSToCSVConverter(DSS_FILE).branch_data)
