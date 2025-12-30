import distopf as opf
import pandas as pd
import numpy as np
from distopf.importer import create_case
from distopf.fbs import fbs_solve
from math import pi

case_csv = create_case(data_path=opf.CASES_DIR / "csv" / "ieee13", start_step=12)
case_cim = create_case(opf.CASES_DIR / "cim/IEEE13.xml")

print("=" * 80)
print("CASE COMPARISON: CSV (ieee13) vs CIM (IEEE13.xml)")
print("=" * 80)

# ============ Branch Data Comparison ============
print("\n" + "=" * 80)
print("BRANCH DATA COMPARISON")
print("=" * 80)

print(f"\nCSV case branches: {len(case_csv.branch_data)}")
print(f"CIM case branches: {len(case_cim.branch_data)}")

print("\nCSV branch types:")
print(case_csv.branch_data["type"].value_counts())
print("\nCIM branch types:")
print(case_cim.branch_data["type"].value_counts())

# Find branches in one case but not the other
csv_names = set(case_csv.branch_data["name"].unique())
cim_names = set(case_cim.branch_data["name"].unique())

only_in_csv = csv_names - cim_names
only_in_cim = cim_names - csv_names

if only_in_csv:
    print(f"\nBranches only in CSV case: {only_in_csv}")
if only_in_cim:
    print(f"\nBranches only in CIM case: {only_in_cim}")

# ============ Bus Data Comparison ============
print("\n" + "=" * 80)
print("BUS DATA COMPARISON")
print("=" * 80)

print(f"\nCSV case buses: {len(case_csv.bus_data)}")
print(f"CIM case buses: {len(case_cim.bus_data)}")

csv_bus_names = set(case_csv.bus_data["name"].unique())
cim_bus_names = set(case_cim.bus_data["name"].unique())

only_in_csv_buses = csv_bus_names - cim_bus_names
only_in_cim_buses = cim_bus_names - csv_bus_names

if only_in_csv_buses:
    print(f"\nBuses only in CSV case: {only_in_csv_buses}")
if only_in_cim_buses:
    print(f"\nBuses only in CIM case: {only_in_cim_buses}")

# Bus load comparison
print("\nCSV bus total loads (kW):")
csv_loads = case_csv.bus_data[["pl_a", "pl_b", "pl_c"]].sum()
print(f"  Phase A: {csv_loads['pl_a']:.3f}")
print(f"  Phase B: {csv_loads['pl_b']:.3f}")
print(f"  Phase C: {csv_loads['pl_c']:.3f}")
print(f"  Total: {csv_loads.sum():.3f}")

print("\nCIM bus total loads (kW):")
cim_loads = case_cim.bus_data[["pl_a", "pl_b", "pl_c"]].sum()
print(f"  Phase A: {cim_loads['pl_a']:.3f}")
print(f"  Phase B: {cim_loads['pl_b']:.3f}")
print(f"  Phase C: {cim_loads['pl_c']:.3f}")
print(f"  Total: {cim_loads.sum():.3f}")

# ============ Generator Data Comparison ============
print("\n" + "=" * 80)
print("GENERATOR DATA COMPARISON")
print("=" * 80)

print(f"\nCSV case generators: {len(case_csv.gen_data)}")
print(f"CIM case generators: {len(case_cim.gen_data)}")

if len(case_csv.gen_data) > 0:
    print("\nCSV Generators:")
    print(case_csv.gen_data[["name", "pa", "pb", "pc"]])
else:
    print("\nCSV: No generators")

if len(case_cim.gen_data) > 0:
    print("\nCIM Generators:")
    print(case_cim.gen_data[["name", "pa", "pb", "pc"]])
else:
    print("\nCIM: No generators")

# ============ Capacitor Data Comparison ============
print("\n" + "=" * 80)
print("CAPACITOR DATA COMPARISON")
print("=" * 80)

print(f"\nCSV case capacitors: {len(case_csv.cap_data)}")
print(f"CIM case capacitors: {len(case_cim.cap_data)}")

if len(case_csv.cap_data) > 0:
    print("\nCSV Capacitors:")
    print(case_csv.cap_data)
else:
    print("\nCSV: No capacitors")

if len(case_cim.cap_data) > 0:
    print("\nCIM Capacitors:")
    print(case_cim.cap_data)
else:
    print("\nCIM: No capacitors")

# ============ Regulator Data Comparison ============
print("\n" + "=" * 80)
print("REGULATOR DATA COMPARISON")
print("=" * 80)

print(f"\nCSV case regulators: {len(case_csv.reg_data)}")
print(f"CIM case regulators: {len(case_cim.reg_data)}")

if len(case_csv.reg_data) > 0:
    print("\nCSV Regulators:")
    print(case_csv.reg_data)
else:
    print("\nCSV: No regulators")

if len(case_cim.reg_data) > 0:
    print("\nCIM Regulators:")
    print(case_cim.reg_data)
else:
    print("\nCIM: No regulators")

# ============ Battery Data Comparison ============
print("\n" + "=" * 80)
print("BATTERY DATA COMPARISON")
print("=" * 80)

print(f"\nCSV case batteries: {len(case_csv.bat_data)}")
print(f"CIM case batteries: {len(case_cim.bat_data)}")

# ============ Summary ============
print("\n" + "=" * 80)
print("SUMMARY OF KEY DIFFERENCES")
print("=" * 80)

differences = []

if len(case_csv.branch_data) != len(case_cim.branch_data):
    differences.append(
        f"Branch count: CSV={len(case_csv.branch_data)}, CIM={len(case_cim.branch_data)}"
    )

if len(case_csv.bus_data) != len(case_cim.bus_data):
    differences.append(
        f"Bus count: CSV={len(case_csv.bus_data)}, CIM={len(case_cim.bus_data)}"
    )

if len(case_csv.gen_data) != len(case_cim.gen_data):
    differences.append(
        f"Generator count: CSV={len(case_csv.gen_data)}, CIM={len(case_cim.gen_data)}"
    )

if len(case_csv.cap_data) != len(case_cim.cap_data):
    differences.append(
        f"Capacitor count: CSV={len(case_csv.cap_data)}, CIM={len(case_cim.cap_data)}"
    )

if len(case_csv.reg_data) != len(case_cim.reg_data):
    differences.append(
        f"Regulator count: CSV={len(case_csv.reg_data)}, CIM={len(case_cim.reg_data)}"
    )

csv_total_load = csv_loads.sum()
cim_total_load = cim_loads.sum()
if abs(csv_total_load - cim_total_load) > 0.001:
    differences.append(
        f"Total load: CSV={csv_total_load:.3f} kW, CIM={cim_total_load:.3f} kW"
    )

if differences:
    for i, diff in enumerate(differences, 1):
        print(f"{i}. {diff}")
else:
    print("No significant differences found")
