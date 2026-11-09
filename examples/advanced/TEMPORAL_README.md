"""
Temporal Decomposition Examples README

This directory contains examples demonstrating the use of TENAPP (Temporal 
Alternating/Approximate/ADMM-based Multi-period OPF) algorithms for solving 
multi-period optimal power flow problems using temporal decomposition.

## Background

Temporal decomposition breaks a large multi-period OPF problem into smaller 
per-time-step subproblems, which are solved iteratively while coordinating 
through battery state-of-charge (SOC) constraints. This approach is useful for:

- Large time horizons (many hours/days)
- Limited computational resources
- Distributed optimization
- Privacy-preserving solutions

## Files

### 1. run_temporal_quick.py
**Simplest example - 6 lines of solving code**

- Loads IEEE 123-bus network with 6-hour horizon
- Sets realistic energy prices
- Solves with TENAPP-1O algorithm
- Prints results

**Run with:**
```bash
uv run examples/advanced/run_temporal_quick.py
```

**Use when:** You just want to get started quickly

### 2. run_temporal_tenapp.py
**Comprehensive example - All three algorithms in detail**

- Loads multi-period case (4 time steps)
- Configures battery parameters
- Runs all three algorithms:
  - TENAPP-1O (first-order method)
  - TENAPP-APRX (approximate dual method)
  - TENAPP-ADMM (ADMM-based method)
- Generates detailed comparison CSV
- Extracts and displays battery dispatch
- Shows convergence iteration histories

**Run with:**
```bash
uv run examples/advanced/run_temporal_tenapp.py
```

**Output files:**
- `scratch/temporal_results/temporal_results_summary.csv` - Algorithm comparison
- `scratch/temporal_results/tenapp_*o_iterations.csv` - Iteration details per algorithm
- `scratch/temporal_results/*.csv` - Full results

**Use when:** You want to understand algorithm behavior and compare performance

### 3. run_temporal_comparison.py
**Advanced example - Flexible comparison framework**

- Creates configurable multi-period cases
- Sets dynamic pricing signals with peak hours
- Runs comparison with custom parameters
- Generates detailed analysis:
  - Cost breakdown
  - Convergence metrics
  - Battery scheduling
  - Iteration timings
- Saves all results to CSV for further analysis

**Run with:**
```bash
uv run examples/advanced/run_temporal_comparison.py
```

**Key features:**
- Modular functions for reuse
- Configurable number of time steps
- Dynamic pricing scenarios
- Comprehensive spreadsheet outputs

**Use when:** You want to run parameter studies and custom scenarios

## Core Algorithms

### TENAPP-1O (First-Order)
```python
from distopf.distributed.temporal import solve_tenapp_1o
result = solve_tenapp_1o(case, objective=energy_cost_min, max_iterations=20)
```

**Pros:**
- Simple dual-based coordination
- Direct interpretation of price signals
- Good for reactive dispatch

**Cons:**
- May need many iterations
- Sensitive to convergence tolerance

### TENAPP-APRX (Approximate Dual)
```python
from distopf.distributed.temporal import solve_tenapp_aprx
result = solve_tenapp_aprx(case, objective=energy_cost_min, alpha2=0.33, alpha3=0.33)
```

**Pros:**
- Gradient-based dual estimates
- Good practical performance
- Tunable approximation accuracy

**Cons:**
- Hyperparameters (alpha2, alpha3) may need tuning
- Approximation may not be exact

### TENAPP-ADMM (ADMM-based)
```python
from distopf.distributed.temporal import solve_tenapp_admm
result = solve_tenapp_admm(case, objective=energy_cost_min, weight=1e2)
```

**Pros:**
- Provable convergence under certain conditions
- Natural dual variable updates
- Adaptable penalty weights

**Cons:**
- Requires careful weight selection
- More computationally intensive per iteration

## Common Output Fields

All algorithms return a dictionary with:

```python
result = {
    "models": [list of LinDistMPL per-time-step models],
    "results": [list of OptimizeResult per time step],
    "iterations": int,                        # Iterations to convergence
    "converged": bool,                        # Whether algorithm converged
    "total_cost": float,                      # Sum of per-time-step costs ($)
    "solve_time": float,                      # Total solve time (seconds)
    "iteration_summaries": pd.DataFrame,      # Iteration-by-iteration details
}
```

## Common Parameters

```python
solve_tenapp_*(
    case,                          # distopf.Case object (n_steps > 1)
    objective=energy_cost_min,     # Objective function
    max_iterations=100,            # Max coordination iterations
    tolerance=1e-3,                # Convergence tolerance
    parallel=False,                # Parallel per-time-step solves
    cost_curve=np.array([...]),    # Hourly electricity prices ($/MWh)
    solver="CLARABEL",             # CVXPY solver (CLARABEL, SCS, MOSEK, etc.)
)
```

## Setting Up a Case

```python
import distopf as opf
import numpy as np

# Create multi-period case
case = opf.create_case(
    opf.CASES_DIR / "csv" / "ieee123_30der_bat",
    n_steps=6,                    # 6-hour horizon
    start_step=0,
)

# Configure voltage limits
case.bus_data.v_max = 1.05
case.bus_data.v_min = 0.95

# Configure generator control mode
case.gen_data.control_variable = "PQ"  # Active and reactive power

# Configure battery parameters
case.bat_data.control_variable = "P"   # Active power control only
case.bat_data["start_soc"] = 0.5       # 50% initial SOC
case.bat_data["max_soc"] = 0.9         # 90% max SOC

# Define hourly electricity costs
cost_curve = np.array([50, 50, 80, 90, 70, 45])  # $/MWh for each hour
```

## Analyzing Results

```python
# Check convergence
if result["converged"]:
    print(f"Solved in {result['iterations']} iterations")
    print(f"Total cost: ${result['total_cost']:.2f}")
else:
    print(f"Did not converge after {result['iterations']} iterations")

# View iteration history
print(result["iteration_summaries"])

# Extract battery dispatch from first time step
model_t0 = result["models"][0]
optim_result_t0 = result["results"][0]
p_batt = model_t0.get_p_batt(optim_result_t0.x)
soc = model_t0.get_soc(optim_result_t0.x)
```

## Performance Tips

1. **Start small**: Begin with short horizons (4-6 hours) before scaling up
2. **Set tolerances carefully**: Looser tolerance (1e-2) = faster, stricter (1e-5) = more accurate
3. **Choose algorithm wisely**: 
   - 1O for first quick solve
   - APRX for practical implementations
   - ADMM for convergence guarantees
4. **Adjust weights**: ADMM weight scaling affects convergence speed
5. **Use realistic pricing**: Include peak hours and time-of-use rates
6. **Monitor iterations**: Check if converging monotonically or oscillating

## Troubleshooting

**Algorithm doesn't converge:**
- Increase `max_iterations`
- Loosen `tolerance` (use 1e-2 instead of 1e-4)
- Check if case is feasible with `opf.run_opf(case, objective="loss_min")`

**Very slow iterations:**
- Disable solver verbosity in CVXPY
- Use `parallel=True` if multi-core available
- Try different solver (SCS faster on some problems, MOSEK more robust)

**Inconsistent results:**
- Check battery efficiency settings
- Verify cost curve values are reasonable
- Ensure schedules/loads are properly formatted

## References

For detailed algorithm descriptions, see parent documentation in:
- `distopf.distributed.temporal.solve_tenapp_1o`
- `distopf.distributed.temporal.solve_tenapp_aprx`
- `distopf.distributed.temporal.solve_tenapp_admm`

"""
