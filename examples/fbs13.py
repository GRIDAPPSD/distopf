import distopf as opf
import pandas as pd
import numpy as np
from distopf.api import create_case
from distopf import (
    plot_voltages,
    plot_gens,
    # plot_network,
    plot_polar,
)
from distopf.fbs import fbs_solve
from math import pi

case = create_case(data_path=opf.CASES_DIR / "csv" / "ieee13", start_step=12)
case.gen_data.control_variable = ""
fbs_results = fbs_solve(case)
i_ang = fbs_results["current_angles"]
v = fbs_results["voltages"]
cur = fbs_results["currents"]
v_ang = fbs_results["voltage_angles"]
cur.index = cur.id
i_ang.index = i_ang.id
v_ang.index = v_ang.id
v.index = v.id
v_phasor = v.loc[:, ["id", "name", "t"]].copy()
v_phasor.loc[:, ["a", "b", "c"]] = v.loc[:, ["a", "b", "c"]] * np.exp(1j * np.radians(v_ang.loc[:, ["a", "b", "c"]]))

i_phasor = cur.loc[:, ["fb", "id", "from_name", "name", "t"]].copy()
i_phasor.loc[:, ["a", "b", "c"]] = cur.loc[:, ["a", "b", "c"]] * np.exp(1j * np.radians(i_ang.loc[:, ["a", "b", "c"]]))

s = cur.loc[:, ["fb", "id", "from_name", "name", "t"]].copy()
s.loc[:, ["a", "b", "c"]] = v_phasor.loc[:, ["a", "b", "c"]] * np.conj(i_phasor.loc[:, ["a", "b", "c"]])

print("Voltages:")
print(v)
print("Voltage Angles:")
print(v_ang)
print("Voltage Phasors:")
print(v_phasor)
print("\nCurrents:")
print(cur)
print("Current Angles:")
print(i_ang)
print("Current Phasors:")
print(i_phasor)
print("\nApparent Power:")
print(s)
