"""
Built-in test case data for DistOPF.

This module provides access to bundled test networks for testing and examples.

Constants
---------
CASES_DIR : pathlib.Path
    Path to the directory containing test case data

Available Cases
---------------
CSV format (CASES_DIR / "csv" / name):
- ieee13 : IEEE 13-bus test feeder
- ieee34 : IEEE 34-bus test feeder
- ieee123 : IEEE 123-bus test feeder
- ieee123_30der : IEEE 123-bus with 30 DERs
- ieee13_battery : IEEE 13-bus with battery storage
- ieee123_30der_bat : IEEE 123-bus with 30 DERs and battery
- 9500 : Large 9500-node network

OpenDSS format (CASES_DIR / "dss" / name):
- ieee13_dss : IEEE 13-bus in OpenDSS format
- ieee123_dss : IEEE 123-bus in OpenDSS format
- ieee9500_dss : IEEE 9500-node in OpenDSS format

Example
-------
>>> from distopf import create_case, CASES_DIR
>>> case = create_case(CASES_DIR / "csv" / "ieee123_30der")
>>> print(f"Buses: {len(case.bus_data)}")
>>> print(f"Branches: {len(case.branch_data)}")
>>> print(f"Generators: {len(case.gen_data)}")

Notes
-----
Cases with "_bat" or "_battery" suffix include battery storage data and are
suitable for multi-period optimization.
"""

from pathlib import Path

CASES_DIR = Path(__path__[0])
