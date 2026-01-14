# DistOPF Parallel Task Queue

This directory contains self-contained task files for LLM agents to work on independently.

## Task Status

| Task | Description | Priority | Status | Files Modified |
|------|-------------|----------|--------|----------------|
| [001](TASK_001_validate_case.md) | Implement Case._validate_case() | High | 🟢 Done | importer.py |
| [002](TASK_002_export_fbs_solve.md) | Export fbs_solve from main module | High | 🟢 Done | __init__.py |
| [003](TASK_003_update_examples.md) | Update examples to new Case API | Medium | 🟢 Done | examples/*.py |
| [004](TASK_004_module_docstrings.md) | Add module docstrings | Low | 🟢 Done | various __init__.py |
| [005](TASK_005_objective_aliases.md) | Add objective function aliases | Medium | 🟢 Done | distOPF.py, __init__.py |

## Status Legend

- 🟡 **Ready** - Available for an agent to pick up
- 🔵 **In Progress** - An agent is working on this
- 🟢 **Done** - Completed and merged
- 🔴 **Blocked** - Waiting on another task

## Parallel Work Rules

### Safe to Parallelize (no file conflicts)

These tasks modify different files and can run simultaneously:

| Agent 1 | Agent 2 | Agent 3 |
|---------|---------|---------|
| Task 001 (importer.py) | Task 003 (examples/) | Task 004 (submodule __init__.py) |

### Sequential Dependencies

These tasks modify the same files and should NOT run in parallel:

- **Task 002** and **Task 005** both modify `__init__.py` and `distOPF.py`
- Run Task 002 first (simpler), then Task 005

### Recommended Parallel Batches

**Batch 1** (can run simultaneously):
- Task 001: Implement validation
- Task 003: Update examples  
- Task 004: Add docstrings

**Batch 2** (after Batch 1):
- Task 002: Export fbs_solve
- Task 005: Add objective aliases

## How to Claim a Task

1. Change status from 🟡 to 🔵 in this README
2. Add your session/agent ID as a comment in the task file
3. Work on the task
4. Run tests: `uv run pytest -m "not slow" --no-cov -q`
5. Change status to 🟢 when complete

## Creating New Tasks

Use this template:

```markdown
# Task XXX: [Title]

**Status:** 🟡 READY FOR WORK  
**Priority:** High/Medium/Low  
**Estimated Effort:** XX minutes  
**Files to Modify:** `path/to/file.py`  
**Tests to Add:** `tests/test_xxx.py`

---

## Problem
[Description of the issue]

## Implementation
[Step-by-step guide]

## Test Cases
[Required tests]

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Notes for Agent
[Any helpful context]
```

## After Completing All Tasks

Once all tasks are done:
1. Update `.github/API_IMPROVEMENT_ISSUES.md` with completion status
2. Run full test suite: `uv run pytest --no-cov`
3. Update `.github/copilot-instructions.md` if patterns changed
---

## ✅ All Tasks Complete (January 13, 2026)

All 5 tasks have been completed and verified:
- **141 tests passing**
- Branch `feature/api-improvements` ready for merge to `main`