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

### Optional Extras

| Extra | Install command | Purpose |
|-------|----------------|---------|
| `cim` | `pip install distopf[cim]` | CIM XML file import support (requires pre-release packages `cim-graph` and `gridappsd-python`) |

> **Note:** The `cim` extra depends on pre-release packages. If you do not need CIM file support, a plain
> `pip install distopf` is sufficient. CIM functionality will raise an informative `ImportError` if those
> packages are not installed.

## Developer Installation
To install the latest version from github:
 
1. From the directory you want to keep your DistOPF files, run:

`git clone https://github.com/nathantgray/distopf.git`

3. Create or activate the python environment you want to use.
4. From the directory where the DistOPF package is stored, run:

`pip install -e .`

To also install the optional CIM extra in editable mode:

`pip install -e ".[cim]"`

This installs your local DistOPF package the python environment you activated. The `-e` option enables editable 
mode, which allows you to directly edit the package and see changes immediately reflected in your environment 
without reinstalling. 


# Getting Started
## Using provided cases:
### Unconstrained Power Flow
```python
import distopf as opf
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123")
result = case.run_pf()
result.plot_network().show(renderer="browser")
```
### DER Curtailment Minimization
```python
import distopf as opf
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
result = case.run_opf(objective="curtail_min", control_variable="P", v_max=1.05, v_min=0.95, gen_mult=10)
result.plot_network().show(renderer="browser")
```

## Optimization Wrappers

DistOPF supports multiple optimization wrappers for solving OPF problems:

### Pyomo Wrapper — LinDistFlow (default)
The default Pyomo wrapper uses the LinDistFlow model (`formulation="lindist"`).
- **Model**: Linear approximation of power flow equations
- **Solver**: Pyomo with linear solvers
- **Speed**: Fast, suitable for real-time applications
- **Accuracy**: Good for systems with small voltage deviations

```python
import distopf as opf
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
result = case.run_opf(wrapper="pyomo", objective="loss")  # formulation="lindist" is the default
```

### Pyomo Wrapper — BranchFlow
The BranchFlow model type uses nonlinear power flow equations with IPOPT or MINLP solvers for higher accuracy.
Use `formulation="branchflow"`.
- **Model**: Nonlinear power flow equations (exact)
- **Solver**: IPOPT (continuous) or MINLP using Gurobi if installed (discrete controls)
- **Speed**: Slower than linear, but more accurate
- **Accuracy**: Nonlinear power flow representation

#### Continuous Optimization (IPOPT)
For continuous optimization without discrete controls:

```python
import distopf as opf
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
result = case.run_opf(
    formulation="branchflow",
    objective="loss",
    solver="ipopt",
)
```

#### Discrete Controls (MINLP)
Enable regulator tap optimization and capacitor switching with MINLP solvers:

```python
import distopf as opf
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123")
result = case.run_opf(
    wrapper="pyomo",
    formulation="branchflow",
    objective="loss",
    control_regulators=True,      # Enable regulator tap control
    control_capacitors=True,       # Enable capacitor switching
    initialize="fbs",              # Recommended for discrete controls
    solver="gurobi",               # MINLP compatible solver 
)
```

### Matrix BESS Wrapper (Multi-Period with Batteries)
The `matrix_bess` wrapper supports multi-period (time-series) optimization with battery energy storage.

```python
import distopf as opf
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der_bat")
result = case.run_opf(wrapper="matrix_bess", objective="loss")
```

#### Wrapper Comparison
| Feature | Matrix/Matrix BESS | Pyomo  |
|---------|---|---|
| Model Type | LinDistFlow (linear) | LinDistFlow or Non-Linear BranchFlow |
| Formulation | Matrix-based | Algebraic equations |
| Solver API | Scipy or CVXPY | Pyomo |
| Prefered Solver | HiGHs or Clarabel | IPOPT, Gurobi, Knitro |

#### Solver Requirements
- **IPOPT**: Install via `conda install -c conda-forge ipopt`. On Ubuntu, `apt-get install coinor-libipopt-dev` only installs headers and shared libraries; it does not provide the `ipopt` executable that Pyomo's `SolverFactory("ipopt")` expects.


### Result Fields

`PowerFlowResult` uses descriptive field names. Short aliases are also accepted for backward compatibility.

| Field Name | Alias |
|---|---|
| `active_power_flows` | `p_flows` |
| `reactive_power_flows` | `q_flows` |
| `active_power_generation` | `p_gens` |
| `reactive_power_generation` | `q_gens` |
| `active_power_loads` | `p_loads` |
| `reactive_power_loads` | `q_loads` |
| `capacitor_reactive_power` | `q_caps` |
| `battery_active_power` | `p_bats` |
| `voltage_magnitudes` | `voltages` |

#### Dual Variables
When running with `duals=True`, dual variables are accessible on the result object:

```python
result = case.run_opf(wrapper="pyomo", objective="loss", duals=True)
result.dual_power_balance_p
result.dual_power_balance_q
result.dual_voltage_drop
result.dual_voltage_limits
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
   -bat_data.csv
```
```python
import distopf as opf
case = opf.create_case(
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
bat_data = pd.read_csv("path/to/your_model_directory/bat_data.csv", header=0)
schedules = pd.read_csv("path/to/your_model_directory/schedules.csv", header=0)  # Optional for multi-period cases
case = opf.Case(
    branch_data=branch_data,
    bus_data=bus_data,
    gen_data=gen_data,
    cap_data=cap_data,
    reg_data=reg_data,
    bat_data=bat_data,
    schedules=schedules
)

```

> **Phase naming convention:** Three-phase buses and lines use phases `a`, `b`, `c`.
> Triplex (North American split-phase residential secondary) buses and lines use phases
> `s1` and `s2`, which are the two 120 V legs of a center-tapped single-phase
> transformer. `s1s2` refers to the 240 V line-to-line connection across both legs.

### branch_data.csv

- fb: From bus id number
- tb: To bus id number
- r_aa, r_ab, r_ac, r_bb, r_bc, r_cc: upper-diagonal resistance matrix elements (p.u.) for 3-phase lines
- x_aa, x_ab, x_ac, x_bb, x_bc, x_cc: upper-diagonal reactance matrix elements (p.u.) for 3-phase lines
- r_s1s1, r_s1s2, r_s2s2: resistance matrix elements (p.u.) for triplex (split-phase) lines
- x_s1s1, x_s1s2, x_s2s2: reactance matrix elements (p.u.) for triplex (split-phase) lines
- primary_phase: primary-side phase for center-tap transformer branches (e.g. "a", "b", "c")
- s_a_max, s_b_max, s_c_max: per-phase apparent power limits (VA)
- type: overhead_line, switch, transformer, center_tap_xfmr, etc.
- name: other name of line
- status: (for switches) OPEN or CLOSED
- s_base: base VA
- v_ln_base: base line-to-neutral voltage
- z_base: base impedance
- phases: phases present on the line (e.g. "abc", "s1s2", "a")

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
- pl_s1, ql_s1, pl_s2, ql_s2, pl_s1s2, ql_s1s2: active and reactive loads for triplex (split-phase) buses (p.u.)
- primary_phase: primary-side phase for triplex buses (e.g. "a", "b", "c")
- phases: phases at bus (e.g. "abc", "a", "ab", "s1s2", etc.)

### gen_data.csv

- id: bus id
- name: generator name
- p_a, p_b, p_c: active power output (p.u.)
- q_a, q_b, q_c: reactive power output (p.u.)
- s_base: base power (VA)
- s_a_max, s_b_max, s_c_max: rated maximum apparent power output per 3-phase (VA)
- p_s1, p_s2: active power output for triplex (split-phase) generators (p.u.)
- q_s1, q_s2: reactive power output for triplex generators (p.u.)
- s_s1_max, s_s2_max: rated maximum apparent power for triplex phases (VA)
- phases: generator phases (e.g. "abc", "s1s2") (this IS implemented)
- q_a_max, q_b_max, q_c_max: (not implemented) maximum reactive power output (p.u.)
- q_a_min, q_b_min, q_c_min: (not implemented) minimum reactive power output (p.u.)

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

## Case and run_opf Options

### `create_case()` / `Case.__init__()` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_path` | required | Path to CSV directory, `.dss` file, or `.xml` CIM file |
| `start_step` | `0` | Starting time step for multi-period analysis |
| `n_steps` | `1` | Number of time steps (`1` = single-period) |
| `delta_t` | `1.0` | Hours per time step (for battery energy calculations) |
| `ignore_schedule` | `False` | If True, ignore schedule data; use multiplier 1.0 everywhere |
| `ignore_gen` | `False` | If True, remove all generators |
| `ignore_bat` | `False` | If True, remove all batteries |
| `ignore_cap` | `False` | If True, remove all capacitors |
| `ignore_reg` | `False` | If True, remove all regulators |

When constructing `Case` directly, pass DataFrames instead of `data_path`:
`branch_data`, `bus_data`, `gen_data`, `cap_data`, `reg_data`, `bat_data`, `schedules`.

### `case.modify()` parameters

| Parameter | Description |
|-----------|-------------|
| `v_swing` | Override substation voltage (scalar or 3-element array, p.u.) |
| `v_min` | Override all bus voltage lower limits (p.u.) |
| `v_max` | Override all bus voltage upper limits (p.u.) |
| `gen_mult` | Scale all generator outputs and ratings |
| `load_mult` | Scale all loads |
| `cvr_p` | CVR factor for active power: `cvr_p = (dP/P)/(dV/V)`. ZIP equivalent: `2kz + ki` |
| `cvr_q` | CVR factor for reactive power: `cvr_q = (dQ/Q)/(dV/V)`. ZIP equivalent: `2kz + ki` |

# OpenDSS Interface
You may also run using an OpenDSS model file as input.

```python
import distopf as opf
case = opf.create_case(
    data_path="path/to/your_model_directory/model.dss",
)
```

# CIM XML Interface

To load a power system model from a CIM XML file, install the optional `cim` extra first:

```
pip install distopf[cim]
```

Then pass the path to your `.xml` file:

```python
import distopf as opf
case = opf.create_case(data_path="path/to/model.xml")
```

If the `cim` extra is not installed, calling `create_case` with a `.xml` file will raise:

```
ImportError: CIM file support requires optional dependencies.
Install them with: pip install distopf[cim]
```

# Citing this tool

Gray, Nathan T., Dubey, Anamika, Reiman, Andrew P., "DistOPF: Advanced Solutions for Distribution Optimal Power Flow Analysis - DistOPF v0.2 Documentation," (2025), https://doi.org/10.2172/2999990 
```
@techreport{osti_2999990,
  author       = {Gray, Nathan T. and Dubey, Anamika and Reiman, Andrew P. and Sadnan, Rabayet},
  title        = {DistOPF: Advanced Solutions for Distribution Optimal Power Flow Analysis - DistOPF v0.2 Documentation},
  institution  = {Pacific Northwest National Laboratory (PNNL), Richland, WA (United States)},
  doi          = {10.2172/2999990},
  url          = {https://www.osti.gov/biblio/2999990},
  place        = {United States},
  year         = {2025},
  month        = {03}}


```
R. Sadnan, N. Gray, A. Bose, A. Dubey and K. P. Schneider, "Scaling Distributed Optimal Renewable Energy Coordination in Unbalanced Distribution Systems," in IEEE Transactions on Sustainable Energy, vol. 17, no. 1, pp. 3-15, Jan. 2026, doi: 10.1109/TSTE.2024.3492976.
```
@ARTICLE{10745555,
  author={Sadnan, Rabayet and Gray, Nathan and Bose, Anjan and Dubey, Anamika and Schneider, Kevin P.},
  journal={IEEE Transactions on Sustainable Energy}, 
  title={Scaling Distributed Optimal Renewable Energy Coordination in Unbalanced Distribution Systems}, 
  year={2026},
  volume={17},
  number={1},
  pages={3-15},
  doi={10.1109/TSTE.2024.3492976}}
```