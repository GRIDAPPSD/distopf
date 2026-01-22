"""
OpenDSS to CSV converter.

This module converts OpenDSS (.dss) files to the CSV format used by DistOPF.
OpenDSS is EPRI's open-source distribution system simulator.

Main Classes
------------
DSSToCSVConverter : Convert OpenDSS model to DistOPF CSV format

Supported Elements
------------------
- Line (with LineCode or geometry)
- Transformer (2-winding)
- RegControl (voltage regulators)
- Capacitor (fixed and switched)
- Load (various models)
- Generator
- PVSystem
- Storage

Example
-------
>>> from distopf.dss_importer import DSSToCSVConverter
>>> converter = DSSToCSVConverter("Master.dss")
>>> # Access converted data
>>> branch_data = converter.branch_data
>>> bus_data = converter.bus_data
>>> gen_data = converter.gen_data

Notes
-----
- Requires `opendssdirect.py` package
- Import is lazy-loaded to avoid slow startup
- Some OpenDSS features not fully supported (see limitations)

Limitations
-----------
- Only 2-winding transformers supported
- Some load models approximated
- Mutual coupling between lines not supported

See Also
--------
distopf.cim_importer : CIM XML format converter
distopf.create_case : High-level case creation (auto-detects format)
"""

from .dss_to_csv_converter import DSSToCSVConverter, load_dss_model

__all__ = ["DSSToCSVConverter", "load_dss_model"]
