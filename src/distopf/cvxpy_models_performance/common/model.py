"""Array-based model container for the vectorized CVXPY models.

Differences from ``distopf.cvxpy_models.common.model``:

- Variables are plain ``cp.Variable`` arrays. Every variable has one axis per
  ``IndexSet`` in the order given to ``CvxModel.add_var`` and (unless
  ``time=False``) a final time axis of length ``n_steps``. For example ``v2`` has
  shape ``(len(bus_phase_set), n_steps)``.
- Constraints are vectorized CVXPY constraints stored in a ``ConstraintRecord``
  together with the row keys (without time) they cover, so that duals can be
  reported with the same layout as the scalar models.

Typical constraint code selects rows, then builds one expression::

    rows, keys = select_rows(m.gen_device_phase_set, is_controlled)
    add_constraint_on_rows(m, "gen_p_limits", (rows, keys), lambda r: m.p_gen[r, :] >= 0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator

import cvxpy as cp
import numpy as np
import scipy.sparse as sp


class IndexSet:
    """Ordered, unique keys with ``pos``: the row of each key in an array axis."""

    def __init__(self, items: Iterable[Any] = ()) -> None:
        self._items = list(dict.fromkeys(items))
        self.pos = {item: i for i, item in enumerate(self._items)}

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item: Any) -> bool:
        return item in self.pos

    def __getitem__(self, position: int) -> Any:
        return self._items[position]

    def indices(self, items: Iterable[Any]) -> np.ndarray:
        """Rows of ``items`` (all must be members) as an integer array."""
        return np.fromiter((self.pos[item] for item in items), dtype=int)

    def __repr__(self) -> str:
        return f"IndexSet({self._items!r})"


def sparse(
    rows: Iterable[int], cols: Iterable[int], vals: Iterable[float], shape: tuple
) -> sp.csr_matrix:
    """Build a CSR matrix, summing duplicate entries."""
    return sp.csr_matrix(
        (
            np.asarray(list(vals), dtype=float),
            (np.asarray(list(rows), dtype=int), np.asarray(list(cols), dtype=int)),
        ),
        shape=shape,
    )


def column(values: Any) -> np.ndarray:
    """Reshape a per-row vector to ``(n, 1)`` so it broadcasts over time."""
    return np.asarray(values, dtype=float).reshape(-1, 1)


@dataclass
class VarSpec:
    """Index information needed to turn a variable array back into a table."""

    name: str
    variable: cp.Variable
    axes: tuple
    time: bool

    @property
    def boolean(self) -> bool:
        return bool(self.variable.attributes["boolean"])


@dataclass
class ConstraintRecord:
    """Vectorized constraint pieces and the row keys they cover.

    ``rows`` lists the index tuples of the rows (first axis) without time. A
    ranged constraint stores two pieces ``[lower, upper]``; the reported dual is
    ``dual(lower) - dual(upper)``.
    """

    name: str
    pieces: list = field(default_factory=list)
    rows: list = field(default_factory=list)


class CvxModel:
    """Attribute container for sets, parameters, variables, and constraints."""

    def __init__(self) -> None:
        self.var_specs: dict[str, VarSpec] = {}
        self.constraint_records: dict[str, ConstraintRecord] = {}
        self.objective_expression = None
        self.problem = None

    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any:
            """Components are attached dynamically; declare that to type checkers."""
            ...

        def __setattr__(self, name: str, value: Any) -> None: ...

    def add_var(
        self,
        name: str,
        *axes: IndexSet,
        time: bool = True,
        nonneg: bool = False,
        boolean: bool = False,
        value: Any = None,
    ) -> cp.Variable:
        """Create ``model.<name>`` with one axis per set and a trailing time axis."""
        shape = tuple(len(axis) for axis in axes)
        if time:
            shape = (*shape, len(self.time_set))
        variable = cp.Variable(shape, nonneg=nonneg, boolean=boolean, name=name)
        if value is not None and variable.size:
            variable.value = np.broadcast_to(
                np.asarray(value, dtype=float), shape
            ).copy()
        self.var_specs[name] = VarSpec(name, variable, tuple(axes), time)
        setattr(self, name, variable)
        return variable

    def set_values(self, name: str, values: Any) -> None:
        """Warm start variable ``name`` from an array broadcastable to its shape."""
        variable = self.var_specs[name].variable
        variable.value = np.broadcast_to(
            np.asarray(values, dtype=float), variable.shape
        ).copy()

    def at(self, name: str, key: Any, t: Any = None) -> Any:
        """Value of variable ``name`` at row ``key`` (and time ``t``), for inspection."""
        spec = self.var_specs[name]
        row = spec.axes[0].pos[key]
        value = spec.variable.value
        if value is None:
            return None
        return value[row] if t is None else value[row, self.time_set.pos[t]]

    @property
    def constraint_names(self) -> list[str]:
        return list(self.constraint_records)

    @property
    def var_names(self) -> list[str]:
        return list(self.var_specs)

    def all_constraints(self) -> list:
        return [
            piece
            for record in self.constraint_records.values()
            for piece in record.pieces
        ]


def add_constraint(
    model: CvxModel, name: str, constraint: Any, rows: Iterable[Any] = ()
) -> ConstraintRecord:
    """Register ``constraint`` (one vectorized constraint or a list) as ``name``.

    ``rows`` are the row keys the constraint covers, in row order, without time.
    An empty list registers the name with no pieces so that the set of
    constraint names does not depend on the data.
    """
    if name in model.constraint_records:
        raise ValueError(f"Constraint {name!r} already exists on this model")
    pieces = constraint if isinstance(constraint, (list, tuple)) else [constraint]
    kept = []
    for piece in pieces:
        if isinstance(piece, bool):
            if piece:
                continue
            raise ValueError(f"Constraint {name!r} is trivially infeasible")
        kept.append(piece)
    record = ConstraintRecord(name, kept, list(rows))
    model.constraint_records[name] = record
    setattr(model, name, record)
    return record


def add_ranged_constraint(
    model: CvxModel,
    name: str,
    lower: Any,
    expression: Any,
    upper: Any,
    rows: Iterable[Any] = (),
) -> ConstraintRecord:
    """Register ``lower <= expression <= upper``; either bound may be ``None``."""
    pieces = []
    if lower is not None:
        pieces.append(expression >= lower)
    if upper is not None:
        pieces.append(expression <= upper)
    return add_constraint(model, name, pieces, rows)


def select_rows(index_set: IndexSet, keep: Iterable[bool]) -> tuple[np.ndarray, list]:
    """Rows of ``index_set`` where ``keep`` is true, and the matching keys."""
    rows = np.flatnonzero(np.asarray(list(keep), dtype=bool))
    return rows, [index_set[i] for i in rows]


def add_constraint_on_rows(
    model: CvxModel,
    name: str,
    selection: tuple[np.ndarray, list],
    build: Callable[[np.ndarray], Any],
) -> ConstraintRecord:
    """Register ``build(rows)`` for the selected rows, or an empty record if none."""
    rows, keys = selection
    if len(rows) == 0:
        return add_constraint(model, name, [], [])
    return add_constraint(model, name, build(rows), keys)
