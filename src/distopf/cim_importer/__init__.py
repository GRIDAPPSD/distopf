"""
CIM (Common Information Model) to CSV converter.

This module converts CIM XML files to the CSV format used by DistOPF.
CIM is an IEC standard (IEC 61970/61968) for power system data exchange.

Main Classes
------------
CIMToCSVConverter : Convert CIM XML to DistOPF CSV format

Supported Equipment
-------------------
- ACLineSegment (overhead/underground lines)
- PowerTransformer (2-winding transformers)
- RatioTapChanger (voltage regulators)
- LinearShuntCompensator (capacitor banks)
- EnergyConsumer (loads)
- SynchronousMachine (generators)
- PhotovoltaicUnit (solar DERs)
- BatteryUnit (storage)

Example
-------
>>> from distopf.cim_importer import CIMToCSVConverter
>>> converter = CIMToCSVConverter("model.xml")
>>> data = converter.convert()
>>> # data contains: branch_data, bus_data, gen_data, cap_data, reg_data

Notes
-----
- Requires `cimgraph` package for CIM parsing
- Import is lazy-loaded to avoid slow startup
- Limited validation coverage - verify output carefully

See Also
--------
distopf.dss_importer : OpenDSS format converter
distopf.create_case : High-level case creation (auto-detects format)
"""

from .cim_to_csv_converter import CIMToCSVConverter, load_cim_model

__all__ = [
    "CIMToCSVConverter",
    "load_cim_model",
]
