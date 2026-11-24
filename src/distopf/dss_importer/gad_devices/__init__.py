"""Per-device-type OpenDSS extractors for the GAD converter.

Each module has one ``extract_<device>(conv)`` function returning the GAD table
for that device type. ``conv`` is the ``GadDSSToCSVConverter`` instance.
"""
