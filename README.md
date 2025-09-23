# DistOPF

DistOPF provides an open-source, multi-phase, unbalanced, optimal power flow (OPF) tool for distribution
systems to aid students and researchers. The tool aids users by providing:

- Unbalanced multi-Phase OPF model generators usable with common Python solver packages such as CVXPY and SciPy;

- A platform for creating and benchmarking new algorithms on a set of standard test systems;

- An OpenDSS model importer allowing users to import power system models directly from the OpenDSS model format;

- Model validation with OpenDSS;

- Functions for visualizing results.


The tool is composed of four major parts, 1) model input system, 2) optimization model formulation, 3) OPF solver
interface, and 4) solution output and visualization. Models are described using a set of CSV files that are read in as
Pandas DataFrames. Buses are described in one CSV having columns for loads, base units, and voltage limits. Lines,
switches, and transformers are described in a CSV having columns for each term in the upper diagonal impedance matrix.
Regulators, capacitor banks, and generators each have their own CSV. To aid in model creation and validation, models can
also be created using OpenDSS and converted to the tabular format. The tool provides classes and functions to make it
easy to formulate and solve the power system for new users while being flexible for advanced users to create new models
and algorithms.
The tool has been used to solve a variety of problems including, conservation voltage reduction, power loss
minimization, and generation curtailment minimization, where either generator real or reactive power injections are controlled.

# Installation

## pip install
```
pip install distopf
```
## Developer Installation
To install the latest version from github:
 
1. From the directory you want to keep your DistOPF files, run:

`git clone https://github.com/nathantgray/distopf.git`

3. Create or activate the python environment you want to use.
4. From the directory where the DistOPF package is stored, run:

`pip install -e .`

This installs your local DistOPF package the python environment you activated. The `-e` option enables editable 
mode, which allows you to directly edit the package and see changes immediately reflected in your environment 
without reinstalling. 


# Getting Started
## Using provided cases:
### Unconstrained Power Flow
```python
import distopf as opf
case = opf.DistOPFCase(data_path="ieee123")
case.run_pf()  # run an unconstrained power flow
case.plot_network().show(renderer="browser")
```
### DER Curtailment Minimization
```python
import distopf as opf
case = opf.DistOPFCase(
    data_path="ieee123_30der",  
    control_variable="P",  # Control DER active power injection
    objective_function="curtail_min",  # DER Curtailment Minimization
    v_max=1.05, v_min=0.95, 
    gen_mult=10  # multiply generator output by 10
)
case.run()
case.plot_network().show(renderer="browser")
```
## Using a custom model.
Create CSVs formatted as shown below and store them in a single folder. The csv names must match exactly as shown. 
Column order is not important. 
```
-your_model_directory
   -branch_data.csv
   -bus_data.csv
   -gen_data.csv
   -cap_data.csv
   -reg_data.csv
```
```python
import distopf as opf
case = opf.DistOPFCase(
    data_path="path/to/your_model_directory",
)
```
Or load them as dataframes

```python
import distopf as opf
import pandas as pd
branch_data = pd.read_csv("path/to/your_model_directory/branch_data.csv", header=0)
bus_data = pd.read_csv("path/to/your_model_directory/bus_data.csv", header=0)
gen_data = pd.read_csv("path/to/your_model_directory/gen_data.csv", header=0)
cap_data = pd.read_csv("path/to/your_model_directory/cap_data.csv", header=0)
reg_data = pd.read_csv("path/to/your_model_directory/reg_data.csv", header=0)
case = opf.DistOPFCase(
    branch_data=branch_data,
    bus_data=bus_data,
    gen_data=gen_data,
    cap_data=cap_data,
    reg_data=reg_data
)

```

### branch_data.csv

- fb: From bus id number
- tb: To bus id number
- r: resistance in p.u.
- x: reactance in p.u.
- type: overhead_line, switch, transformer, etc.
- name: other name of line
- status: (for switches) OPEN or CLOSED
- s_base: base VA
- v_ln_base: base line-to-neutral voltage
- z_base: base impedance

### bus_data.csv

- id: unique id for each bus (integer starting at 1)
- name: bus name
- pl_a, ql_a, pl_b, ql_b, pl_c, ql_c: active and reactive loads p.u.
- bus_type: SWING or PQ. SWING bus is voltage source
- v_a, v_b, v_c: voltage magnitude p.u. (input parameter for SWING bus. Other not used as input)
- v_ln_base: base line-to-neutral voltage (V)
- s_base: base power (VA)
- v_min, v_max: voltage magnitude limits (p.u.)
- cvr_p, cvr_q: conservation voltage reduction parameters; alternative to ZIP model for voltage dependant loads. (set to
  0 for no voltage dependence)
- phases: phases at bus (e.g. "abc", "a", "ab", etc.)

### gen_data.csv

- id: bus id
- name: generator name
- pa, pb, pc: active power output (p.u.)
- qa, qb, qc: reactive power output (p.u.)
- s_base: base power (VA)
- sa_max, sb_max, sc_max: rated maximum apparent power output (VA)
- phases: generator phases (abc string) (this IS implemented)
- qa_max, qb_max, qc_max: (not implemented) maximum reactive power output (p.u.)
- qa_min, qb_min, qc_min: (not implemented) minimum reactive power output (p.u.)

### cap_data.csv

- id: bus id
- name: capacitor name
- q_a, q_b, q_c: nominal reactive power (p.u.)
- phases: capacitor phases (abc string)

### reg_data.csv

- fb: From bus id number
- tb: To bus id number
- name: regulator name 
- tap_a, tap_b, tap_c: tap position (p.u.) -16 to +16; 0 is no tap change

## DistOPFCase Options
```
    Use this class to create a distOPF case, run it, and save and plot results.
    Parameters
    ----------
    config: str or dict
        Path to JSON config or dictionary with parameters to create case. Alternative to using **config.
    data_path: str or pathlib.Path
        Path to the directory containing the data CSVs or path to OpenDSS model. Will also accept names of
        cases include in package e.g. "ieee13", "ieee34", "ieee123".
    output_dir: str or pathlib.Path
        (default: "output") Directory to save results.
    branch_data : pd.DataFrame or None
        DataFrame containing branch data (r and x values, limits). Overrides data found from data_path.
    bus_data : pd.DataFrame or None
        DataFrame containing bus data (loads, voltages, limits). Overrides data found from data_path.
    gen_data : pd.DataFrame or None
        DataFrame containing generator/DER data. Overrides data found from data_path.
    cap_data : pd.DataFrame or None
        DataFrame containing capacitor data. Overrides data found from data_path.
    reg_data : pd.DataFrame or None
        DataFrame containing regulator data. Overrides data found from data_path.
    v_swing: Number or size-3 array
        Override substation voltage. Scalar or 3-phase array. Per Unit.
    v_min: Number
        Override all voltage minimum limits. Per Unit.
    v_max: Number
        Override all voltage maximum limits. Per Unit.
    gen_mult: Number
        Scale all generator outputs and ratings. Per Unit.
    load_mult:
        Scale all loads.
    cvr_p:
        CVR factor for voltage dependent loads. Active power component. cvr_p = (dP/P)/(dV/V)
        To convert from ZIP parameters, kz, ki, kp: cvr_p = 2kz + 1ki
    cvr_q:
        CVR factor for voltage dependent loads. Reactive power component.cvr_q = (dQ/Q)/(dV/V)
        To convert from ZIP parameters, kz, ki, kp: cvr_q = 2kz + 1ki
    control_variable: str
        Control variable for optimization. Options (case-insensitive):
            None: Power flow only with no optimization. `objective_function` options will be ignored.
            "P": Active power injections from generators. Active power outputs set in gen_data.csv will be ignored
                 and reactive power outputs set in gen_data static.
            "Q": Reactive power injections from generators.
                 Active power outputs set in gen_data.csv are constant and reactive power outputs set in
                 gen_data.csv will be ignored.
    objective_function: str or Callable
        Objective function for optimization. Options (case-insensitive):
            "gen_max": Maximize output of generators. Uses scipy.optimize.linprog.
            "load_min": Minimize total substation active power load. Uses scipy.optimize.linprog.
            "loss_min": Minimize total line active power losses. Quadratic. Uses CVXPY.
            "curtail_min": Minimize DER/Generator curtailment. Quadratic. Uses CVXPY.
            "target_p_3ph": Substation load tracks active power target on each phase. Quadratic. Uses CVXPY.
            "target_q_3ph": Substation load tracks reactive power target on each phase. Quadratic. Uses CVXPY.
            "target_p_total": Substation load tracks total active power. Quadratic. Uses CVXPY.
            "target_q_total": Substation load tracks total reactive power. Quadratic. Uses CVXPY.
    show_plots: bool
        (default False) If true, renders plots in browser
    save_results: bool
        (default False) If true, saves result data to CSVs in output_dir
    save_plots: bool
        (default False) If true, saves interactive plots as html to output folder
    save_inputs: bool
        (default False) If true, saves model CSV and other input parameters.
        NOTE CSVs include any modifications made by other parameters such as gen_mult, load_mult, v_max, v_min, or
        v_swing.
```

# OpenDSS Interface
You may also run using an OpenDSS model file as input.

```python
import distopf as opf
case = opf.DistOPFCase(
    data_path="path/to/your_model_directory/model.dss",
)
```

# Citing this tool
Paper coming soon.