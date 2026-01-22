# Examples

Some examples are Marimo notebooks. To use these, run `marimo edit`. Marimo will
open the browser and list the available notebooks.

---

DistOPF Example Suite
====================

This directory contains a comprehensive set of simple examples showing how to use DistOPF
for various optimal power flow (OPF) problems.

Start Here
----------

**01_simple_power_flow.py** (1 min read)
  The absolute simplest example. Load a network, run power flow, plot results.
  - Shows the basic Case API
  - Demonstrates power flow analysis (no optimization)
  - Introduces the plotting capabilities

**07_quick_start.py** (30 seconds)
  Your first OPF in under 2 minutes of code.
  - Minimal viable example
  - Copy this as a template for your own work

Basic Concepts
--------------

**02_loss_minimization.py** (5 min)
  Most common OPF objective: minimize real power losses.
  - Shows loss minimization OPF
  - Demonstrates DER active and reactive power control
  - Compares results before/after optimization

**03_voltage_optimization.py** (5 min)
  Minimize voltage deviations for grid stability.
  - Shows voltage deviation objective
  - Demonstrates reactive power control
  - Useful for voltage regulation

**04_comparing_backends.py** (10 min)
  Run the same problem with different solvers.
  - Matrix backend (fast, convex solver)
  - Pyomo backend (flexible, can handle nonlinear)
  - Shows solver comparison

Advanced Topics
---------------

**05_control_strategies.py** (10 min)
  Different ways to control DERs in the network.
  - Fixed output vs. optimized control
  - Active power only (P) vs. reactive (Q) vs. both (PQ)
  - Per-generator control strategies
  - How control capabilities affect grid operation

**08_reactive_power_importance.py** (5 min)
  Why reactive power matters for distribution networks.
  - Compares P-only vs. PQ control
  - Shows voltage regulation improvement
  - Demonstrates the importance of modern DERs

**09_optimization_tradeoffs.py** (10 min)
  Different objectives lead to different results.
  - Loss minimization (economic focus)
  - Voltage minimization (safety focus)
  - Trade-off analysis between objectives

**11_fbs_power_flow.py** (5 min)
  Forward-Backward Sweep (FBS) power flow solver for radial networks.
  - Demonstrates FBS solver usage via `case.run_fbs()` and regulator modeling
  - Compares FBS results with matrix-based `case.run_pf()` for validation
  - Useful for fast 3-phase unbalanced power flow and validation

Visualization & Analysis
------------------------

**06_visualization_gallery.py** (10 min)
  Tour of all visualization capabilities.
  - Network plot with voltage heatmap
  - Voltage profile plot
  - Power flow visualization
  - Generator output plot
  - How to interpret each visualization

**10_working_with_results.py** (10 min)
  How to work with OPF results.
  - Access voltage and power flow DataFrames
  - Perform post-optimization analysis
  - Export results to CSV
  - Common analysis patterns

Quick Comparison Table
======================

| Example | Time | Focus | What You'll Learn |
|---------|------|-------|-------------------|
| 01_simple_power_flow | 1 min | Basics | Load network, run power flow, plot |
| 07_quick_start | 30 sec | Template | Minimal working OPF code |
| 02_loss_minimization | 5 min | Objective | Most common OPF use case |
| 03_voltage_optimization | 5 min | Objective | Alternative optimization goal |
| 04_comparing_backends | 10 min | Solvers | Different optimization engines |
| 05_control_strategies | 10 min | Control | How DER control affects results |
| 08_reactive_power_importance | 5 min | Physics | Why Q matters in distribution |
| 09_optimization_tradeoffs | 10 min | Analysis | Comparing different objectives |
| 11_fbs_power_flow | 5 min | FBS | Forward-Backward Sweep power flow and comparison |
| 06_visualization_gallery | 10 min | Plotting | All available plots |
| 10_working_with_results | 10 min | Analysis | Working with results |

How to Run Examples
===================

Each example is standalone and can be run independently:

    uv run examples/01_simple_power_flow.py
    uv run examples/02_loss_minimization.py
    # ... etc
    uv run examples/11_fbs_power_flow.py

All examples produce plots that open in your browser.

Key Concepts
============

**Network (Case)**
  A distribution network with buses, lines, transformers, and loads.
  Load from CSV or create programmatically.

**Objective**
  The goal of optimization:
  - "loss_min": Minimize real power losses
  - "voltage_dev": Minimize voltage deviations

**Control Variable**
  What DERs (generators, batteries, inverters) are allowed to control:
  - "": Fixed output (no control)
  - "P": Control active power only
  - "Q": Control reactive power only
  - "PQ": Control both active and reactive power

**Backend (Solver)**
  Which optimization engine to use:
  - "matrix": CVXPY + CLARABEL (fast, convex problems)
  - "pyomo": Pyomo + IPOPT (flexible, nonlinear problems)

**Results**
  After solving, you get:
  - voltages: Voltage magnitude/angle at each bus
  - power_flows: Active and reactive power on each line
  - p_gens: Active power from each generator
  - q_gens: Reactive power from each generator

Common Workflows
================

**Scenario 1: I want to minimize losses in my network**
  -> Use 02_loss_minimization.py as template
  -> Set control_variable="PQ" for maximum flexibility
  -> Use backend="matrix" for speed

**Scenario 2: I need better voltage regulation**
  -> Use 03_voltage_optimization.py as template
  -> Focus on reactive power control (Q or PQ)
  -> Compare with 08_reactive_power_importance.py

**Scenario 3: I want to understand different control strategies**
  -> Study 05_control_strategies.py
  -> Modify control_variable per generator
  -> Compare results

**Scenario 4: Which solver should I use?**
  -> Check 04_comparing_backends.py
  -> For convex problems: matrix backend
  -> For nonlinear constraints: pyomo backend

**Scenario 5: I need to analyze the results**
  -> Study 10_working_with_results.py
  -> Extract voltages, power flows, generator outputs
  -> Perform statistical analysis
  -> Export to CSV for further processing

Resources
=========

For more information:
- README.md: Project overview
- src/distopf/backends/base.py: Backend API documentation
- Copilot instructions: Development guidelines
- Unit tests (tests/): More detailed examples

Tips & Tricks
=============

1. **Fastest first run**: Start with 01_simple_power_flow.py (no optimization, just PF)

2. **Template for your work**: Copy 07_quick_start.py and modify network name

3. **Debug control strategies**: Use print(case.gen_data['control_variable']) to verify

4. **Compare backends**: Run same problem with both "matrix" and "pyomo" backends

5. **Export results**: Use result.voltages.to_csv() to save for Excel analysis

6. **Batch analysis**: Run same case multiple times with different control strategies

7. **Interactive debugging**: Add print statements before plotting to understand results