# DSS Converter: Old vs. New

Compares `src/distopf/dss_importer/dss_to_csv_converter_old.py` (~1950 lines) with the current `dss_to_csv_converter.py` (~2100 lines). The public class is `DSSToCSVConverter` in both. The diff is close to a full rewrite, so this is organized by topic rather than by hunk.

Findings come from reading both files. Items marked **(bug fix)** corrected behavior that was wrong in the old converter. Items marked **(feature)** are new capability. Items marked **(refactor)** change structure only. Numerical equivalence was not re-run for this document.

## Summary

| Area | Old | New |
|---|---|---|
| Voltage bases | `Bus.kVBase()` plus ad-hoc `sqrt(3/4)` and `2/sqrt(3)` fixes for secondaries | Explicit nominal base per bus, propagated through lines/reactors and across transformers by nameplate ratio. User overrides allowed. |
| Center-tap transformer | Diagonal 2x2, no primary leg, no validation | Full 2x2 star-equivalent matrix including primary leg, with validation |
| Line charging | Ignored | `b_a`, `b_b`, `b_c` columns |
| Loads | Solved `CktElement.Powers()` | Nominal kW/kvar, CVR-aware, delta loads, load shapes |
| Generators | Fixed `gen_shape="PV"` | Real shape names and schedule columns |
| Capacitors | Only 1-phase or `abc` | Any phase subset, kV-ratio scaling |
| Regulators | Hard-coded 0.00625 step | Step taken from `MinTap`/`MaxTap`/`NumTaps`, new columns |
| Flow tables | Three near-identical 100-line loops | One shared `_pd_branch_table` |
| Topology | `@cache`d methods, recomputed ad hoc | Built once in `_refresh()` |
| Output | CSVs only | Adds `schedules.csv`, `get_power_flow_result()` |

## 1. Voltage bases and per-unit conversion

**Old.** Each place that needed a base called `Bus.kVBase()` and patched it locally:

- Triplex lines: `kVBase * sqrt(3/4)`
- Center-tap transformers: `kVBase * 2/sqrt(3)`
- Bus data: `kVBase * sqrt(3/4)` for secondary buses
- `get_v_solved`: multiplied secondary voltages by `2/sqrt(3)`

These corrections are not equivalent (`sqrt(3/4)` against `2/sqrt(3)`, which are different numbers). They also assume OpenDSS reports the secondary as `V_LL/sqrt(3)`, which depends on the source/transformer data. Per-unit impedance on a branch could therefore be inconsistent with the bus `v_ln_base` the OPF uses. **(bug fix)**

**New.** `_build_nominal_voltage_bases()` computes one base per bus:

1. Buses are grouped into line/reactor-connected components. Transformers separate components.
2. Each component is anchored by, in order: an explicit `voltage_bases` override, the source `BasekV` (divided by sqrt(3) for three phases), or a center-tap winding rating.
3. Unanchored components get `upstream base * nameplate kV ratio` (`_propagate_bases_through_transformers`). The per-unit turns ratio then equals the tap.
4. Conflicting anchors in one component raise `ValueError`. Unknown override names raise.
5. Components that cannot be reached keep the DSS reporting base.

All branches are normalized on the receiving bus base (`_branch_row`), and `get_v_solved` divides by the same base. No fixed correction factor remains. **(bug fix, feature)**

New constructor argument: `voltage_bases: dict[str, float]` (line-to-neutral volts).

## 2. Center-tap (split-phase) transformers

**Old** `append_center_tap_transformers`:

- Used only windings 2 and 3 as `r_s1s1`/`r_s2s2` and `x_s1s1`/`x_s2s2`. The off-diagonal terms were hard-coded to 0.
- The primary winding's star-equivalent impedance was computed in the OpenDSS math but never used.
- `Xhl/Xht/Xlt` were read before `Wdg()` selection with no check on winding order or taps.
- Impedances were not on a per-winding kVA base. Reactances used winding 2's base. The star-circuit formulas assume all share the winding 1 base.

**New** `_read_center_tap_parameters`:

- Validates: 3 windings, `phases=1`, shared secondary bus, `.1.0`/`.0.2` in either order, no delta, positive kV/kVA, unity taps, equal secondary voltages. Each failure raises with the transformer name.
- Handles primary connections written `.1`, `.1.0`, or without nodes.
- Computes star reactances from `Xhl/Xht/Xlt` on the winding 1 kVA base. Each winding `%R` is converted using its own kVA base. All are referred to the secondary voltage.
- Exports the full coupled matrix `[[zp+z1, -zp], [-zp, zp+z2]]`, including the primary leg and mutual coupling. **(bug fix)**
- `append_center_tap_transformers` checks the turns ratio against the chosen voltage bases (FBS copies the primary pu voltage to both legs) and that the edge runs primary to secondary.

## 3. Triplex lines

- Old: assumed `RMatrix()` always had 4 elements (`reshape(2, 2)`). A 1-conductor secondary line would crash. New: reads `n` phases, validates `n in (1, 2)`, requires matching conductor nodes at both ends, checks the matrix is symmetric, and maps actual conductor order to s1/s2. **(bug fix)**
- Old exported `phases="s1s2"` for every triplex line. New exports the actual legs. **(bug fix)**
- Open triplex lines are now skipped. Old always exported them as `CLOSED` on a base computed from `kVBase`. **(bug fix)**
- New checks that both endpoint voltage bases match.

## 4. Primary lines, reactors, transformers

- **Line Z matrix** (`_get_line_zmatrix`): old guessed three-phase-ness from the string `bus1.split(".")` length (4 parts or 1 part) and used a nested counter for the rest. New parses nodes once, drops neutral nodes, and embeds the `n x n` matrix into the 3x3 with `np.ix_`. This also handles cases like `bus.1.2.3.0`.
- **Line charging (feature)**: new `_get_line_susceptance` computes per-phase shunt susceptance from `CMatrix` (including negative mutual terms and frequency), exported as `b_a`, `b_b`, `b_c`. Old ignored line capacitance.
- **Transformer Z** (`_get_transformer_zmatrix`): new scales `z_base` by `Tap()^2`, matching the OpenDSS definition of `%R/%X` on the tapped winding voltage. Old ignored taps. **(bug fix)**
- **Switches (feature)**: lines with `IsSwitch()` are exported as `type="switch"`.
- **Phases**: new `_branch_phases` reads the phase letters from the terminal at the oriented from-bus. Old did this too but with string comparisons and a redundant branch.
- **Open elements**: handled in one place (`_is_open`, `_pd_edges`). Old `_build_secondary_buses` only honored open status for lines that were switches, while `get_bus_names` honored it for every element, so the two topologies could differ. **(bug fix)**

## 5. Topology and caching

- **Old**: `@cache` on `get_bus_names`, `bus_names_to_index_map` and `_bus_depth_map`. `functools.cache` on methods keys on `self`, so after `update()` re-solved the circuit the cached bus list and depth map were stale. The same edge-collection loop appeared in three methods. **(bug fix, refactor)**
- **New**: `_refresh()` rebuilds center taps, topology (`bus_names`, `bus_names_to_index_map`, `_bus_depth`), secondary buses, voltage bases, and every table in dependency order. `__init__` and `update()` both call it. Edge collection is `_pd_edges` and spanning-tree construction is `_build_topology`.
- `_orient_edge` is a single sort key (reachable first, shallower first, then DFS index) instead of a chain of `if`s.
- `get_spanning_tree` uses the named constant `_OPEN_EDGE_WEIGHT`. `_discover_neutral_nodes` is new and used by the `PowerFlowResult` export.
- The unused `_get_powers`, `num_phase_map`, `get_bus_names`, and `bus_names_to_index_map_fun` helpers are removed. They are replaced by `_PRIMARY_NODE_PHASE`, `_phases_from_nodes`, and the `bus_names` attribute.

## 6. Loads

**Old** `_get_loads`:

- Took power from `CktElement.Powers()`, the solved terminal power. That already includes voltage dependence, while the OPF applies `cvr_p/cvr_q` again, so the CVR effect was applied twice for non-constant-power loads. The number also changes after a re-solve. **(bug fix)**
- Per-bus `cvr_p/cvr_q` were summed across loads (`"sum"`), so two loads with `cvr=2` gave 4. **(bug fix)**
- Delta loads were read from conductor powers by index.
- Loads with more than one nonzero node were assigned by position (`p_values[phase_index]`), which breaks when a neutral conductor is listed.

**New**:

- Uses nominal `kW`/`kvar`, scaled by `(base/rated)^cvr` when the bus base differs from the load's rated kV (`_rated_to_base_ratio`).
- `cvr_p/cvr_q` per bus are the kW/kvar-weighted average of the loads at that bus (`_cvr_p_kw`/`_kw` ...).
- Delta loads (`IsDelta()`) are converted to line-to-neutral injections using the standard line-to-line to line-to-neutral phasor shift.
- Secondary 240 V loads go to `pl_s1s2`, 120 V loads to one leg.
- CVR lookup is a separate `_load_cvr()` backed by `_LOAD_MODEL_CVR`.

## 7. Load shapes and schedules (feature)

Not present in the old converter. New:

- `schedule_steps` constructor argument (default 96) and a `schedules` DataFrame with a `time` column.
- `bus_data["load_shape"]` and `gen_data["gen_shape"]` reference columns of `schedules`.
- `_shape_values` reads `PMult`/`QMult`, repeats single-point shapes, and raises on mismatched `SInterval` versus `StepSize` or too few points.
- `_blend_shapes` and `_link_load_shapes` produce a kW-weighted blend when several loads at one bus have different shapes. Separate `.p`/`.q` columns are created only when the curves differ.
- `to_csv` now writes `schedules.csv`.

## 8. Generators

- Old always wrote `gen_shape="PV"` regardless of the DSS shape. New uses the PV system's `yearly` shape (lowercased) and links it into `schedules`. Several generators at one bus with different shapes get a blended curve. **(bug fix, feature)**
- Phase handling is shared between primary and secondary paths via `assign(phase, share)`. 240 V secondary generators split equally across s1/s2; 120 V generators use one leg. Old repeated those assignments longhand.
- An old secondary generator with unrecognized nodes silently produced no phases. New produces an empty row with the same effect but through one clear branch (`legs = []`).
- Column list and aggregation come from module constants (`_GEN_COLUMNS`, `_GEN_FIRST_COLUMNS`, `_gen_agg()`). **(refactor)**
- `update_gen_p/q` share `_update_gens`. **(refactor)**

## 9. Capacitors

- Old: single-phase case branches on `cap_phase != "abc"`. Two-phase capacitors (`.1.2`) took the single-phase branch and got zero `q_*`, because `cap_phase in {"a"}` fails for `"ab"`. **(bug fix)** New divides total kvar evenly over any phase subset.
- New scales `q` by `(bus base / rated kV)^2`, so capacitors whose rated kV differs from the bus base keep the correct pu value. Secondary buses use ratio 1. **(bug fix)**

## 10. Regulators

- Old: `tap = (ratio - 1) / 0.00625`, only for transformers listed in a `RegControl`, and missing values were filled with 1. New uses the actual step `(MaxTap - MinTap) / NumTaps` from the DSS transformer, so non-standard regulators (e.g. 16 steps of 0.00625 versus 32 steps of 0.005) are correct. **(bug fix)**
- New also includes transformers whose second winding tap is not 1, even without a `RegControl`.
- New adds `ratio_min`, `ratio_max`, `num_taps` columns to `reg_data`. Missing taps are filled with 0 (neutral) rather than 1. **(feature)**
- Tap sign is flipped when `_orient_edge` reverses the DSS terminal order (old behavior retained).

## 11. Bus data

- `v_ln_base` now comes from the nominal base map (see section 1).
- Bus ids are looked up by name from the shared map; old used `enumerate` position for nothing and then re-looked up.
- `load_shape` column added and defaulted to `""`.
- Phases come from `_bus_phases` (nodes minus neutral, secondary legs mapped to s1/s2, else `_phases_from_nodes`). Old used `num_phase_map[str(Bus.Nodes())]`, which raised `KeyError` for any node list not in the table, such as `[1, 2, 3, 0]` or a bus with a ground node. **(bug fix)**
- Column order puts `id` and `name` first.

## 12. Solved voltages, currents, and flows

- **`get_v_solved`**: old merged three per-phase tables, applied the `2/sqrt(3)` correction, and moved a/b into s1/s2. New reads `Bus.VMagAngle()` per bus and maps nodes to a/b/c or s1/s2, dividing by the nominal base. The index is the bus id, and unconnected phases are NaN.
- **`get_currents`, `get_current_angles`, `get_apparent_power_flows`**: old had three copies of the same ~100-line loop. New has one `_pd_branch_table(raw, extract, dtype)` and three small `extract` callbacks. **(refactor)**
- Disabled or open elements missing from the topology are skipped in the new table walk. In the old one they raised `KeyError` in `bus_names_to_index_map`. **(bug fix)**
- Current base uses `s_base / (kVBase*1000)` in both. Neither uses the nominal-base map for currents.
- `get_p_flows`/`get_q_flows` share `_split_flows`. Missing phases stay NaN.
- Per-unit phase blanking for parallel branches is done once in `_combine_parallel`.

## 13. New API: `get_power_flow_result()`

Not in the old converter. Builds a `distopf.results.PowerFlowResult` from the OpenDSS snapshot, with node voltages, branch powers/currents, and device powers. It takes `t`, `case`, and `solve` arguments and reports solve time. It uses terminal powers at the receiving end and the secondary windings for center-tap transformers, FBS-convention secondary voltage angles, and rejects explicit non-ground neutrals via `_discover_neutral_nodes`.

## 14. Code structure and housekeeping

- Module constants replace repeated literals: `_ABC_Z`, `_SPLIT_Z`, `_Z_COLUMNS`, `_B_COLUMNS`, `_BRANCH_COLUMNS`, `_PHASE_ORDER`, `_PRIMARY_NODE_PHASE`, `_SECONDARY_NODE_LEG`.
- `_branch_row` replaces `_create_branch_row` and the two hand-written dicts (about 30 lines each) in the triplex and center-tap code. It emits all impedance columns, NaN for those that do not apply.
- `get_branch_data` returns columns in `_BRANCH_COLUMNS` order and includes the new `b_*` columns.
- `from __future__ import annotations`, `TYPE_CHECKING` import of `PowerFlowResult`, typed callbacks.
- Class docstring now lists the modeling assumptions (radial, canonical split-phase windings, unity taps, consistent source/winding voltages).
- `to_csv` loops over table names.
- Dead commented-out code in the old file (reactor sub-3-phase block, tap fallbacks) is gone. Sub-3-phase reactors still raise `NotImplementedError`.

## Behavioral differences to expect

Existing exported CSVs will change in these ways:

1. `v_ln_base` and `z_base` for secondary buses and branches follow the nominal base, not `kVBase` with a correction.
2. Center-tap transformer rows include primary-leg impedance and off-diagonal terms.
3. Load `pl_*`/`ql_*` are nominal power (not solved terminal power), and `cvr_p/cvr_q` are averages instead of sums.
4. `gen_shape` is a real shape name or empty, not `"PV"`.
5. New columns: `b_a`, `b_b`, `b_c` (branch), `ratio_min`, `ratio_max`, `num_taps` (reg), `load_shape` (bus), plus `schedules.csv`.
6. Regulator tap counts can differ for regulators with non-default step size, and missing taps are 0 rather than 1.
7. Models that previously loaded now fail early with explicit errors: conflicting voltage bases, center-tap transformers that are non-canonical, delta, or tapped, loadshape interval mismatches.
8. Shunt capacitance now affects OPF results because `b_*` is exported. Downstream models must consume it or ignore the columns.

## Not changed / still limitations

- Radial networks only.
- Reactors must be three-phase.
- Loads with multiple connected buses raise.
- Explicit neutral conductors are not modeled.
- Parallel branches between the same buses are summed, as before.
- `dss_to_csv_converter_old.py` is kept only for reference. Nothing imports it.

## 15. IEEE 9500 validation additions

Validated against OpenDSS on `ieee9500_dss` (no controls; exact FBS replay and LinDist compared with OpenDSS). The investigation log is not kept in the repository.

- **`delta_to_wye` flag** on `DSSToCSVConverter`, `create_case` and `create_case_from_dss` (default `False`). Rewrites every 3-phase transformer winding to `conn=wye`, drops a trailing `.4` neutral, and re-solves, so OpenDSS and DistOPF model the same network. LinDist has no delta model; without it voltage errors were 1.06 pu on B/C. A floating `.4` primary neutral (not grounding it) was also the cause of 1.124 pu at `m1069-mt1` in the hand-made wye case. **(feature)**
- **Open-switch phases**: `_energized_phases` limits bus phases to those carried by non-open branches. Before, a bus behind an open single-phase switch inherited phases B/C from the OpenDSS node list and was solved at about 1.06 pu while OpenDSS had 0. **(bug fix)**
- **Switched-off capacitors**: `get_cap_data` skips capacitors whose states are all off. Before, an off bank (300 kvar/phase at `R20185`) was exported as on, giving a 0.318 pu Q error. **(bug fix)**

Results (delta case with the flag): voltage max diff about 0.01 pu (was 1.06). Q-flow max diff vs OpenDSS: LinDist 0.24/0.29/0.22 (was 0.54-0.69), FBS 0.06 (was about 0.38).

Remaining differences, not importer bugs: transformer no-load/magnetizing losses (about 62 kW, 156 kvar total in OpenDSS) are not modeled, which accounts for the FBS vs OpenDSS gap; LinDist is lossless and ignores charging, which accounts for its larger gap.

## 16. Transformer no-load / magnetizing shunt

- New branch columns `g_sh_a/b/c` (no-load loss) and `b_sh_a/b/c` (magnetizing susceptance), per unit on `s_base`. Written by `_get_transformer_shunt` from OpenDSS `%noloadloss` and `%imag` (percent of kVA, read via `Properties.Value`), split evenly over the transformer phases. Set for regular and center-tap transformers; NaN elsewhere, summed over parallel branches. **(feature)**
- `fbs.py` builds `_shunt_y = g - j*b` at each transformer's from-bus and subtracts `_shunt_y * v` in `_injection` (constant-impedance load, `p = g|V|^2`, `q = b|V|^2`).
- The Pyomo and CVXPY OPF models consume these columns (see section 17). The matrix models do not; their results are unchanged.

Validation (FBS vs OpenDSS, max abs diff in p.u.):

| Case | Q flow before | Q flow after |
|------|---------------|--------------|
| ieee9500 (delta_to_wye) | 0.055-0.059 | 0.024-0.034 |
| EPRI J1 | about 0.25 | about 0.25 (transformers have `%imag` = 0, so no effect) |

On 9500, P flow diff is now 0.5-1.2e-2 and voltage diff 0.6-2.1e-3. The remaining EPRI J1 Q gap (about 0.25 pu at the substation) has another cause and is not yet diagnosed.

### Single-transformer validation

Test feeder: source, 0.2 mi line, 0.01 mi line (switch stand-in), 6 MVA wye-wye transformer (12.47/4.16 kV), 0.05 mi line, 3-phase load. Source impedance is 1e-6 and `%Rs` is equal on both windings, so FBS and OpenDSS agree with no shunt (dV 7.7e-7). All bus voltages are 0.957-1.000 pu. Eight variants sweep `%noloadloss` (0-5), `%imag` (0-10), `Xhl`/`%R`, and load. Errors are FBS minus OpenDSS in p.u. on 1 MVA.

- **Magnitude and sign are correct.** OpenDSS draws exactly `%noloadloss` kW and `%imag` kvar of rated kVA at no load, and the converter columns match.
- **Total power at the source improves by 1-2 orders of magnitude.** Worst cases (5% / 10%): dP 0.29 to 6.9e-3, dQ 0.61 to 6.8e-3 with the shunt modeled.
- **Voltages improve only 5-8%.** Residual dV is 1.2e-3 (`%imag` 2%) to 1.1e-2 (5% / 10%, high Z).
- **Placement does not match OpenDSS.** FBS puts the shunt at the from-bus. OpenDSS behaves as if it sits on the secondary side. Moving it to the to-bus reduces dV to about 5e-7 and branch-flow error to the no-shunt baseline (about 1e-5) when series drop is small. With high series impedance or heavy load, neither end matches (dV 3-4e-3 at the to-bus, branch flow error 0.02-0.08), so the shunt would need an internal node of the transformer.
- **Measure at the source, not at the transformer.** Branch-flow errors at the transformer branch reach 0.2 pu with the from-bus placement, but they largely cancel in total source power (at most 7e-3 for moderate cases).

## 17. Transformer shunt in the OPF models

- **New `TransformerShuntProvider`** in `pyomo_models/devices/transformer_shunt.py`. It reads `g_sh_*`/`b_sh_*` from the branch data, stores `{(from_bus, phase): (g, b)}`, and injects `-g*v2` (P) and `-b*v2` (Q) at the from-bus, i.e. a constant-impedance load. It adds no constraints. NaN or missing values count as 0. **(feature)**
- **Registered** in the LinDist and BranchFlow factories, `capacity_expansion` and `mpssd`. `cvxpy_models` reuses the Pyomo provider; `cvxpy_models_performance/devices/transformer_shunt.py` builds `g`/`b` arrays aligned with `bus_phase_set` and returns `-g*v2`, `-b*v2` as arrays.
- **Validation of the bus**: the provider checks the from-bus phase, not the branch phase, because center-tap shunts sit on the primary phase while the branch phases are `s1s2`. A missing bus-phase raises `ValueError`.
- **Placement**: the shunt is kept at the from-bus, which the converter orients to winding 1 (primary). All 1,275 shunt rows in 9500 and 816 of 817 in EPRI J1 have the primary as `fb`. The swing bus has no balance equation, so a shunt on a branch from the swing bus would be dropped from the OPF; no shunt-carrying branch starts at the swing bus in either case.
- **Tests**: `tests/pyomo_models/test_transformer_shunt.py` (flow into the from-bus rises in P and Q) and `test_transformer_shunt_matches_scalar` in `tests/cvxpy_models_performance/test_devices_parity.py`.
- **Results on 9500**: Pyomo and `cvxpy_models_performance` substation flows match exactly. The shunt adds about 0.02 pu P and 0.06 pu Q at the substation. The remaining LinDist gap to OpenDSS is mostly line losses.
- **Not covered**: the matrix models, the exact BranchFlow model with the new provider, and about 17 (9500) / 3 (EPRI J1) transformers with no shunt.