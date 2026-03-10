# distOPF Benchmarking Suite

This directory contains benchmarking and validation scripts for comparing results across different backends, solvers, and optimization objectives.

## Purpose

The benchmarking suite ensures consistency and identifies regressions when:
- Adding new backends or solvers
- Modifying optimization models or constraints
- Updating solver configurations
- Comparing different formulations (linear vs. nonlinear, single-period vs. multi-period)

## Current Benchmarks

### 1. Pyomo vs Multiperiod (ieee123_30der)

**File:** `compare_pyomo_vs_multiperiod_ieee123_30der.py`

**Description:** Compares the linear OPF results from two different backends on the IEEE 123-bus system with 30 DERs:
- **Pyomo backend:** Uses IPOPT solver with LinDistFlow model
- **Multiperiod backend:** Uses CLARABEL solver with matrix model

**Objective:** Loss minimization

**What it measures:**
- Solver status and termination conditions for each backend
- Voltage magnitudes at each bus (per phase: a, b, c)
- Voltage delta statistics (max, mean, std) between backends
- Objective function values and deltas
- Solve time for each backend

**How to run:**
```bash
uv run python benchmarks/compare_pyomo_vs_multiperiod_ieee123_30der.py
```

**Output:**
- Summary table printed to stdout
- Detailed markdown report: `benchmarks/results/<timestamp>_comparison.md`
- JSON results: `benchmarks/results/<timestamp>_comparison.json`

**Expected behavior:**
- Both backends should solve successfully (status = "ok")
- Voltage deltas should be small (typically < 1e-3 p.u. for well-posed problems)
- Objective values should be close (within a few percent)
- If either backend fails, the benchmark exits with code 1

## Interpreting Results

### Solver Status

- **ok:** Solver converged to a solution
- **warning:** Solver converged but with warnings (e.g., locally infeasible point)
- **infeasible:** Problem is infeasible
- **error:** Solver encountered an error

### Voltage Deltas

Voltage differences between backends are reported as:
- **Max ΔV:** Maximum absolute voltage difference across all buses and phases
- **Mean ΔV:** Average voltage difference
- **Std ΔV:** Standard deviation of voltage differences
- **Per-phase stats:** Separate statistics for phases a, b, c

**Interpretation:**
- Deltas < 1e-4 p.u.: Excellent agreement (expected for convex problems)
- Deltas 1e-4 to 1e-3 p.u.: Good agreement (typical for different solvers/formulations)
- Deltas > 1e-3 p.u.: Investigate potential issues (different problem formulations, solver settings, or numerical issues)

### Objective Deltas

Objective function differences are reported as:
- **Δ Objective:** Absolute difference in objective values
- **Δ Objective (%):** Relative difference as a percentage

**Interpretation:**
- < 0.1%: Excellent agreement
- 0.1% to 1%: Good agreement
- > 1%: Investigate potential issues

## Adding New Benchmarks

To add a new benchmark:

1. Create a new script in `benchmarks/` following the naming convention: `compare_<backend1>_vs_<backend2>_<case>.py`

2. Use the shared comparison library:
   ```python
   from benchmarks.comparison import SolverResult, compare_results, format_comparison_table, format_detailed_report
   ```

3. Implement functions to run each backend and return `SolverResult` objects

4. Use `compare_results()` to compute statistics

5. Save results using `format_comparison_table()` and `format_detailed_report()`

6. Example structure:
   ```python
   def run_backend_1(case) -> SolverResult:
       # Run backend 1, return SolverResult
       pass
   
   def run_backend_2(case) -> SolverResult:
       # Run backend 2, return SolverResult
       pass
   
   def main():
       case = create_case(...)
       result_1 = run_backend_1(case)
       result_2 = run_backend_2(case)
       comparison = compare_results(result_1, result_2)
       # Save and print results
   ```

## Results Archive

Benchmark results are saved in `benchmarks/results/` with timestamps:
- `<timestamp>_comparison.md`: Human-readable markdown report
- `<timestamp>_comparison.json`: Machine-readable JSON results

These can be used to track regressions over time or compare against baseline results.

## Troubleshooting

### Benchmark fails with "One or both backends failed to solve"

This indicates that at least one backend could not find a feasible solution. Possible causes:
1. **Problem is infeasible:** Check case data (voltage limits, power balance, etc.)
2. **Solver configuration issue:** Check solver parameters and availability
3. **Model formulation issue:** Verify constraints and objective function

**Next steps:**
- Run `uv run python examples/pyomo_nlp_comparison.py` to see if FBS power flow works
- Check solver logs for specific error messages
- Verify case data is valid using `case._validate_case()`

### Large voltage deltas between backends

Possible causes:
1. **Different problem formulations:** Linear vs. nonlinear models may have different solutions
2. **Solver precision:** Different solvers have different numerical tolerances
3. **Initialization:** Some solvers benefit from good initial points
4. **Constraint differences:** Verify both backends use the same constraints

**Next steps:**
- Compare individual bus voltages to identify which buses have large deltas
- Check if deltas are consistent across phases
- Verify both backends are solving the same problem (same case, objective, constraints)

### Benchmark runs slowly

- Pyomo with IPOPT can be slow for large problems
- CLARABEL is typically faster for convex problems
- Consider running on a smaller test case first

## Related Files

- `benchmarks/comparison.py`: Shared comparison library
- `tests/test_verify_pyomo.py`: Old verification test (being phased out)
- `tests/test_verify_multiperiod.py`: Old verification test (being phased out)
- `examples/pyomo_nlp_comparison.py`: Example comparing linear vs. nonlinear backends
