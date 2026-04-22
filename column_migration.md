# Column Naming Convention Migration: `pa` → `p_a`

## Progress Notes

- [x] Step 1 complete (2026-04-21): Added backwards-compat rename shims in `src/distopf/utils.py` for generator, capacitor, and branch legacy columns. `handle_*_input()` now normalizes to underscore-based canonical columns.
- [ ] Step 2 in progress: Update Python source files to only reference underscore-based names.
- [ ] Step 3 pending: Update built-in CSV headers.
- [ ] Step 4 pending: Update tests.
- [ ] Step 5 pending: Update examples.

## Goal

Standardize all phase-suffixed column names to use an underscore separator:
`{prefix}_{phase}` for all phases including `a`, `b`, `c`, `s1`, `s2`.

This eliminates the inconsistency where loads (`pl_a`, `ql_a`) already used underscores
but generators (`pa`, `sa_max`), capacitors (`qa`), and branch impedances (`raa`, `xab`)
did not.

## Backwards Compatibility Strategy

The `handle_*_input()` functions in `utils.py` are the single point of entry for all
DataFrames. The migration plan is:

1. **Add rename logic in `utils.py`** — each `handle_*_input()` function detects old
   column names (without underscore) and renames them to the new convention before
   returning. This makes every downstream consumer work with the new names without
   changes, and old CSVs / user-supplied DataFrames continue to work transparently.
2. **Update all Python source** to reference new column names directly (no more
   conditional `if phase in TRIPLEX_PHASES` branching needed).
3. **Update all built-in CSV data files** to use new headers.
4. **Update tests and examples** to use new names.
5. **Remove the compatibility shim** from `utils.py` in a future cleanup once all CSVs
   and user-facing docs are updated.

---

## Column Mapping

### `gen_data.csv`

| Old | New |
|-----|-----|
| `pa` | `p_a` |
| `pb` | `p_b` |
| `pc` | `p_c` |
| `qa` | `q_a` |
| `qb` | `q_b` |
| `qc` | `q_c` |
| `sa_max` | `s_a_max` |
| `sb_max` | `s_b_max` |
| `sc_max` | `s_c_max` |
| `qa_max` | `q_a_max` |
| `qb_max` | `q_b_max` |
| `qc_max` | `q_c_max` |
| `qa_min` | `q_a_min` |
| `qb_min` | `q_b_min` |
| `qc_min` | `q_c_min` |
| `ps1` | `p_s1` |
| `ps2` | `p_s2` |
| `qs1` | `q_s1` |
| `qs2` | `q_s2` |
| `ss1_max` | `s_s1_max` |
| `ss2_max` | `s_s2_max` |
| `ss1s2_max` | `s_s1s2_max` |
| `qs1_max` | `q_s1_max` |
| `qs2_max` | `q_s2_max` |
| `qs1_min` | `q_s1_min` |
| `qs2_min` | `q_s2_min` |

### `cap_data.csv`

| Old | New |
|-----|-----|
| `qa` | `q_a` |
| `qb` | `q_b` |
| `qc` | `q_c` |

### `branch_data.csv`

| Old | New |
|-----|-----|
| `raa` | `r_aa` |
| `rab` | `r_ab` |
| `rac` | `r_ac` |
| `rbb` | `r_bb` |
| `rbc` | `r_bc` |
| `rcc` | `r_cc` |
| `xaa` | `x_aa` |
| `xab` | `x_ab` |
| `xac` | `x_ac` |
| `xbb` | `x_bb` |
| `xbc` | `x_bc` |
| `xcc` | `x_cc` |
| `sa_max` | `s_a_max` |
| `sb_max` | `s_b_max` |
| `sc_max` | `s_c_max` |

> Note: triplex branch columns (`r_s1s1`, `x_s1s2`, etc.) already use underscores —
> no change needed.

### `bus_data.csv`

Load columns (`pl_a`, `ql_a`, `pl_s1`, etc.) already use underscore convention. No
change needed.

---

## Files to Change

### Step 1 — `utils.py` (backwards-compat rename shim + new column names)

**File:** `src/distopf/utils.py`

- `handle_gen_input()`: add rename dict for old→new before returning; update column list
  in the empty-DataFrame branch
- `handle_cap_input()`: add rename for `qa/qb/qc` → `q_a/q_b/q_c`
- `handle_branch_input()`: add rename for `raa/…/xcc` → `r_aa/…/x_cc` and
  `sa_max/sb_max/sc_max` → `s_a_max/s_b_max/s_c_max`

Rename shim pattern:
```python
_GEN_COL_RENAMES = {
    "pa": "p_a", "pb": "p_b", "pc": "p_c",
    "qa": "q_a", "qb": "q_b", "qc": "q_c",
    "sa_max": "s_a_max", "sb_max": "s_b_max", "sc_max": "s_c_max",
    "qa_max": "q_a_max", "qb_max": "q_b_max", "qc_max": "q_c_max",
    "qa_min": "q_a_min", "qb_min": "q_b_min", "qc_min": "q_c_min",
    "ps1": "p_s1", "ps2": "p_s2", "qs1": "q_s1", "qs2": "q_s2",
    "ss1_max": "s_s1_max", "ss2_max": "s_s2_max",
}
gen_data = gen_data.rename(columns={k: v for k, v in _GEN_COL_RENAMES.items() if k in gen_data.columns})
```

---

### Step 2 — Python source files

#### `src/distopf/api.py`
- Lines ~841–842: `["qa_max", "qb_max", "qc_max"]` → `["q_a_max", "q_b_max", "q_c_max"]` etc.
- Lines ~844–845: `["pa", "pb", "pc", "qa", "qb", "qc"]` → `["p_a", "p_b", "p_c", "q_a", "q_b", "q_c"]`
- Lines ~1456–1458: scaling assignments for `pa/pb/pc`, `qa/qb/qc`, `sa_max/sb_max/sc_max`

#### `src/distopf/fbs.py`
- Line ~155–157: `get(gen, "pa", 0)` → `get(gen, "p_a", 0)` for all 6 gen columns
- Line ~170: `get(cap, "qa", 0)` → `get(cap, "q_a", 0)` for all 3 cap columns

#### `src/distopf/validators.py`
- Line ~123: `["sa_max", "sb_max", "sc_max"]` → `["s_a_max", "s_b_max", "s_c_max"]`

#### `src/distopf/plot.py`
- Lines ~1118–1125: `gen_data["pa"]` etc. → `gen_data["p_a"]`
- Line ~1239: `["qa", "qb", "qc"]` → `["q_a", "q_b", "q_c"]`
- Lines ~1249, 1252: `["pa", "pb", "pc"]` and `["qa", "qb", "qc"]`

#### `src/distopf/pyomo_models/lindist.py`
- `_create_rx_parameters`: remove `if phase_pair.startswith("s")` branch; use
  `f"r_{phase_pair}"` / `f"x_{phase_pair}"` for all phase pairs
- `_create_generator_parameters`: replace `f"s{phase}_max"` → `f"s_{phase}_max"`,
  `f"q{phase}_max"` → `f"q_{phase}_max"`, etc.; delete the commented-out triplex block
- `_create_capacitor_parameters`: `f"q{phase}"` → `f"q_{phase}"`
- `_create_branch_thermal_parameters`: `f"s{phase}_max"` → `f"s_{phase}_max"`;
  column checks `sa_max/sb_max/sc_max` → `s_a_max/s_b_max/s_c_max`

#### `src/distopf/pyomo_models/nl_branchflow.py`
- `f"r{phase_pair}"` → `f"r_{phase_pair}"`, `f"x{phase_pair}"` → `f"x_{phase_pair}"`
- `f"s{phase}_max"` → `f"s_{phase}_max"`, `f"q{phase}_max"` etc.
- `f"p{phase}"` → `f"p_{phase}"`, `f"q{phase}"` → `f"q_{phase}"`

#### `src/distopf/pyomo_models/constraints.py`
- Docstrings and any column references: `sa_max/sb_max/sc_max` → `s_a_max/s_b_max/s_c_max`

#### `src/distopf/matrix_models/base.py`
- Lines ~145–158: `branch.raa` → `branch.r_aa`, `branch.rab` → `branch.r_ab`, etc.
  for all 12 impedance attribute accesses
- Lines ~760–770: `"sa_max"` → `"s_a_max"` in column checks; `self.branch.sa_max` →
  `self.branch.s_a_max` etc.
- Anywhere `self.gen[f"q{phase}"]` or `self.cap[f"q{phase}"]` is used

#### `src/distopf/matrix_models/lindist_p_gen.py`
- Line ~92: `self.gen[f"q{phase}"]` → `self.gen[f"q_{phase}"]`
- Line ~122: `self.cap[f"q{phase}"]` → `self.cap[f"q_{phase}"]`
- Lines ~156–158: `self.gen_data.qa` → `self.gen_data.q_a` etc.

#### `src/distopf/matrix_models/lindist_q_gen.py`
- Line ~98: `self.gen[f"p{phase}"]` → `self.gen[f"p_{phase}"]`
- Line ~128: `self.cap[f"q{phase}"]` → `self.cap[f"q_{phase}"]`
- Lines ~138–140: `self.gen_data.pa` → `self.gen_data.p_a` etc.

#### `src/distopf/matrix_models/matrix_bess/base_mp.py`
- Lines ~211–224: `branch.raa` → `branch.r_aa` etc. for all 12 impedance accesses
- Lines ~1166–1178: `"sa_max"` column checks and `self.branch.sa_max` accesses

#### `src/distopf/dss_importer/dss_to_csv_converter.py`
- Lines ~830–841: `raa=`, `rab=`, … `xcc=` kwargs → `r_aa=`, `r_ab=`, … `x_cc=`
- Lines ~957–968, 1094–1105: same pattern in nan-fill dicts
- Lines ~1164–1175: `"raa": "sum"` etc. in aggregation dicts
- Lines ~1295–1317: `pa=0`, `pb=0`, `pc=0`, `qa=0`, `qb=0`, `qc=0`,
  `sa_max=0`, `qa_max=0`, etc. in generator dict construction
- Lines ~1366–1371: `each_gen["qa_max"]` etc.
- Lines ~1378–1400: column name lists and aggregation kwargs
- Lines ~1407–1429: aggregation dicts for `pa/pb/pc`, `qa/qb/qc`, `sa_max` etc.
- Lines ~1492–1540: `qa=`, `qb=`, `qc=` for capacitor data

#### `src/distopf/cim_importer/processors/generator_processor.py`
- Lines ~62–78 and ~179–195: dict keys `"pa"`, `"sa_max"`, `"qa_max"`, etc.
- Lines ~210–213: `gen_data["pa"]` assignments

#### `src/distopf/cim_importer/processors/capacitor_processor.py`
- Line ~26: `"qa": 0.0` dict key
- Line ~72: `cap_data["qa"]` assignment

#### `src/distopf/cim_importer/cim_to_csv_converter.py`
- Lines ~228–246: aggregation dict keys (`"pa"`, `"sa_max"`, `"qa_max"` etc.)
- Lines ~273–282: column name lists
- Lines ~330–338: `gen_df.loc[..., "sa_max"]` etc.

---

### Step 3 — Built-in CSV data files

All need header row updated. Data values are unchanged.

#### `gen_data.csv` (18 files)
All case directories under `src/distopf/cases/csv/`:
- `ieee13/`, `ieee13_battery/`, `ieee123/`, `ieee123_30der/`, `ieee123_30der_bat/`,
  `ieee123_alternate/`, `ieee34/`, `ieee33/`, `9500/`, `9500-primary-network/`,
  `2Bus-1ph-batt/`, `3Bus-1ph-batt/`, `4Bus-YY-Bal_dss/`, `4Bus-YY-Bal_dss_batt/`,
  `minimal_triplex/`, `triplex_pv/`, `triplex_3ph/`, `smartds_small/`

Columns to rename in headers: `pa→p_a`, `pb→p_b`, `pc→p_c`, `qa→q_a`, `qb→q_b`,
`qc→q_c`, `sa_max→s_a_max`, `sb_max→s_b_max`, `sc_max→s_c_max`, `qa_max→q_a_max`,
`qb_max→q_b_max`, `qc_max→q_c_max`, `qa_min→q_a_min`, `qb_min→q_b_min`,
`qc_min→q_c_min`.

For triplex cases (`minimal_triplex/`, `triplex_pv/`, `triplex_3ph/`): also
`ps1→p_s1`, `ps2→p_s2`, `qs1→q_s1`, `qs2→q_s2`, `ss1_max→s_s1_max`,
`ss2_max→s_s2_max`.

#### `cap_data.csv` (17 files)
Same case directories (minus `ieee33/` which has no cap file).
Columns to rename: `qa→q_a`, `qb→q_b`, `qc→q_c`.

#### `branch_data.csv` (18 files)
Same case directories.
Columns to rename: `raa→r_aa`, `rab→r_ab`, `rac→r_ac`, `rbb→r_bb`, `rbc→r_bc`,
`rcc→r_cc`, `xaa→x_aa`, `xab→x_ab`, `xac→x_ac`, `xbb→x_bb`, `xbc→x_bc`,
`xcc→x_cc`.
Where present: `sa_max→s_a_max`, `sb_max→s_b_max`, `sc_max→s_c_max`.

---

### Step 4 — Tests

#### `tests/pyomo_models/test_pyomo_lindist.py`
- Lines ~28–29, 34: inline branch DataFrame: `"raa"` → `"r_aa"`, `"rab"` → `"r_ab"`,
  `"xaa"` → `"x_aa"`
- Lines ~60–74: inline gen DataFrame: all `pa/pb/pc/qa/qb/qc/sa_max/sb_max/qa_max` etc.
- Lines ~85–87: inline cap DataFrame: `"qa"/"qb"/"qc"` → `"q_a"/"q_b"/"q_c"`

#### `tests/test_utilities.py`
- Lines ~53–66: inline gen DataFrame: `pa/pb/pc/qa/qb/qc/sa_max/qa_max/qb_max/qc_max/qa_min` etc.

---

### Step 5 — Examples

#### `examples/optimization/thermal_limits.py`
- Lines ~19–20, 35: `"sa_max"/"sb_max"/"sc_max"` → `"s_a_max"/"s_b_max"/"s_c_max"`

---

## Implementation Order

1. `utils.py` — add backwards-compat shim (enables remaining steps to proceed without
   breaking existing tests at each intermediate step)
2. CSV files — update headers (can be done with a script)
3. Python source files — update column references
4. Tests — update inline DataFrames
5. Examples — update references
6. Remove commented-out triplex branch in `lindist.py` `_create_generator_parameters`
7. (Future) Remove backwards-compat shim from `utils.py` once no old CSVs remain in
   the wild
