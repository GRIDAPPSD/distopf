Date: 2026-04-21
Project: distopf_2-triplex

### 1) What I Did
- Implemented triplex support end-to-end today, from initial pipeline wiring through converter/model/test updates.
- Brought triplex functionality to a working state on the small generated triplex feeders.
- Debugged converter export issues around secondary generator columns and naming consistency.
- Fixed p versus s_max inconsistencies in triplex generator export and validated corrected CSV outputs.
- Ran focused triplex unit/integration tests and confirmed the triplex-focused subset is passing.
- Investigated residual DSS vs OPF mismatches on smartds_small and isolated likely data/representation causes.
- Added a temporary phase-aggregation change to suppress duplicate phase strings, and flagged it for revisit because it can break legitimate multi-phase merge behavior.

### 2) What’s Working (define mostly)
Mostly working means:
- The triplex path works on small generated feeders.
- Current run can solve in your environment when ignore_gen is enabled.
- Core topology/import/model plumbing for triplex is operational in this scoped path.
- Focused triplex tests pass in this scope.

### 3) Blocker / What’s Stuck
- SmartDS with generators enabled still does not solve reliably.
- Generator semantics are unresolved when Pmpp is greater than s_rated (or equivalent AC limit); policy is not finalized.
- Phase aggregation behavior still needs a robust merge strategy; duplicate strings like s1s2s1s2 were suppressed with a stopgap that can break valid merged-phase cases.
- Center-tap transformer power-flow reference convention is unresolved:
  - Current implementation is secondary-referred.
  - OpenDSS comparison is primary-referred.
  - This reference-frame mismatch makes direct validation difficult until a canonical convention or explicit conversion is defined.

### 4) Next Steps / Plan
- Reproduce SmartDS generator-enabled failure deterministically with targeted diagnostics.
- Define and implement a clear policy for Pmpp greater than s_rated handling.
- Replace phase aggregation stopgap with canonical phase-set merge logic that preserves valid multi-phase rows.
- Define canonical center-tap reporting basis (primary or secondary) for validation, and add conversion when needed.
- Re-run SmartDS generator-inclusive parity checks after semantics are aligned.
- Document final comparison assumptions so DSS vs OPF validation is apples-to-apples.

### 5) Status Checkboxes
- [x] Implemented triplex path from start and validated on small generated feeders.
- [x] Confirmed current local run succeeds with ignore_gen enabled.
- [x] Captured key unresolved generator and phase-aggregation issues.
- [ ] SmartDS solve with generators enabled is stable and understood.
- [ ] Final policy for Pmpp greater than s_rated is implemented and tested.
- [ ] Phase aggregation is robust for merged triplex rows.
- [ ] Center-tap primary vs secondary flow reference is standardized for validation.
- [ ] Generator-inclusive DSS vs OPF parity is validated end-to-end on SmartDS assumptions.

Mental note:
Progress is real, and tomorrow should start with semantics alignment (generator limits and center-tap reference basis) before deeper solver-level debugging.
