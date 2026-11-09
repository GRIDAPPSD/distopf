"""Delta loads: converter CSVs replayed with FBS must match OpenDSS."""

import pytest

from distopf import create_case
from distopf.dss_importer.dss_to_csv_converter import DSSToCSVConverter
from distopf.fbs import replay_exact_power_flow_from_opf_result

DELTA_FEEDER = """
clear
Set DefaultBaseFrequency=60
new circuit.DeltaFeeder basekV=12.47 phases=3 bus1=sourcebus pu=1.0 mvasc3=2000000 2000000

new linecode.lc nphases=3 basefreq=60 units=mi
~ rmatrix=(0.3465 | 0.1560 0.3375 | 0.1580 0.1535 0.3414)
~ xmatrix=(1.0179 | 0.5017 1.0478 | 0.4236 0.3849 1.0348)
~ cmatrix=(0 | 0 0 | 0 0 0)

new line.l1 bus1=sourcebus bus2=n1 linecode=lc length=0.5 units=mi
new line.l2 bus1=n1 bus2=n2 linecode=lc length=0.5 units=mi

new load.d_ab bus1=n2.1.2 phases=1 conn=delta kV=12.47 kW=600 kvar=250 model=1 vminpu=0.7
new load.d_bc bus1=n2.2.3 phases=1 conn=delta kV=12.47 kW=200 kvar=80  model=1 vminpu=0.7
new load.d_ca bus1=n1.3.1 phases=1 conn=delta kV=12.47 kW=900 kvar=300 model=1 vminpu=0.7

set voltagebases=[12.47]
calcvoltagebases
solve
"""

TWO_LOADS_ONE_BUS = """
clear
Set DefaultBaseFrequency=60
new circuit.TwoLoads basekV=12.47 phases=3 bus1=sourcebus pu=1.0 mvasc3=2000000 2000000

new linecode.lc nphases=3 basefreq=60 units=mi
~ rmatrix=(0.3465 | 0.1560 0.3375 | 0.1580 0.1535 0.3414)
~ xmatrix=(1.0179 | 0.5017 1.0478 | 0.4236 0.3849 1.0348)
~ cmatrix=(0 | 0 0 | 0 0 0)

new line.l1 bus1=sourcebus bus2=n1 linecode=lc length=0.5 units=mi

! constant-power and constant-impedance loads on the same bus
new load.pq bus1=n1.1 phases=1 kV=7.2 kW=300 kvar=100 model=1 vminpu=0.7
new load.z  bus1=n1.1 phases=1 kV=7.2 kW=100 kvar=40 model=2 vminpu=0.7
! delta load rated line-to-line on a single pair
new load.d  bus1=n1.2.3 phases=1 conn=delta kV=12.47 kW=200 kvar=80 model=2 vminpu=0.7

set voltagebases=[12.47]
calcvoltagebases
solve
"""


def _converter(tmp_path, dss_text):
    dss_file = tmp_path / "feeder.dss"
    dss_file.write_text(dss_text)
    return DSSToCSVConverter(dss_file, s_base=1e6)


def _max_error(converter, tmp_path, attr, key):
    converter.to_csv(tmp_path / "csv")
    case = create_case(tmp_path / "csv")
    converter.dss.Text.Command("Set Controlmode=OFF")
    converter.dss.Solution.Solve()
    expected = converter.get_power_flow_result(t=0)
    actual = replay_exact_power_flow_from_opf_result(case, expected)
    a = getattr(actual, attr).set_index(key)[["a", "b", "c"]].astype(float)
    b = getattr(expected, attr).set_index(key)[["a", "b", "c"]].astype(float)
    idx = a.index.intersection(b.index)
    return float((a.loc[idx] - b.loc[idx]).abs().max().max())


def test_single_pair_delta_loads_match_opendss(tmp_path):
    converter = _converter(tmp_path, DELTA_FEEDER)
    assert _max_error(converter, tmp_path, "voltages", "name") < 1e-4


def test_delta_load_with_line_to_line_kv_is_not_rescaled(tmp_path):
    converter = _converter(tmp_path, TWO_LOADS_ONE_BUS)
    converter.to_csv(tmp_path / "csv")
    case = create_case(tmp_path / "csv")
    bus = case.bus_data.set_index("name")
    # kW is exported nominal, per-unit on s_base=1e6 (kW * 1e3 / s_base)
    assert bus.loc["n1", "pl_b"] + bus.loc["n1", "pl_c"] == pytest.approx(0.2, rel=1e-6)


def test_cvr_exponents_are_load_weighted_average_per_bus(tmp_path):
    converter = _converter(tmp_path, TWO_LOADS_ONE_BUS)
    converter.to_csv(tmp_path / "csv")
    bus = create_case(tmp_path / "csv").bus_data.set_index("name")
    # (300 kW * 0 + 100 kW * 2 + 200 kW * 2) / 600 kW = 1.0, not the sum 4
    assert bus.loc["n1", "cvr_p"] == pytest.approx(1.0, abs=1e-3)


TWO_PHASE_DELTA = """
clear
Set DefaultBaseFrequency=60
new circuit.T basekV=12.47 phases=3 bus1=sourcebus pu=1.0 mvasc3=2000000 2000000
new linecode.lc nphases=3 basefreq=60 units=mi
~ rmatrix=(0.3465 | 0.1560 0.3375 | 0.1580 0.1535 0.3414)
~ xmatrix=(1.0179 | 0.5017 1.0478 | 0.4236 0.3849 1.0348)
~ cmatrix=(0 | 0 0 | 0 0 0)
new line.l1 bus1=sourcebus bus2=n1 linecode=lc length=1 units=mi
new load.x bus1=n1.{nodes} phases=2 conn=delta kV=12.47 kW=2000 kvar=800 model=1 vminpu=0.7
set voltagebases=[12.47]
calcvoltagebases
solve
"""


@pytest.mark.parametrize("nodes", ["1.2.3", "3.1.2"])
def test_two_phase_delta_load_matches_opendss(tmp_path, nodes):
    converter = _converter(tmp_path, TWO_PHASE_DELTA.format(nodes=nodes))
    assert _max_error(converter, tmp_path, "voltages", "name") < 5e-4
