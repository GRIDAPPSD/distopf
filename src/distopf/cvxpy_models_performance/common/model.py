"""Array-based model container for the vectorized CVXPY models.

Differences from ``distopf.cvxpy_models.common.model``:

- Variables are plain ``cp.Variable`` arrays. Every variable has one axis per
  index set, in the order given to ``CvxModel.add_var``, and (unless
  ``time=False``) a final time axis of length ``n_steps``. For example ``v2`` has
  shape ``(len(bus_phase_set), n_steps)``.
- The position of an index along an axis is given by ``IndexSet.pos``.
- Constraints are vectorized CVXPY constraints stored in a
  ``ConstraintRecord`` together with the row keys (without time) they cover, so
  that duals can be reported with the same layout as the scalar models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

_NO_DEFAULT = object()


def _flatten(item: Any) -> tuple:
    return item if isinstance(item, tuple) else (item,)


class IndexSet:
    """Ordered collection of unique indices with O(1) membership and position."""

    def __init__(self, items: Iterable[Any] = (), dimen: int | None = None) -> None:
        self._items = list(dict.fromkeys(items))
        self.pos = {item: i for i, item in enumerate(self._items)}
        if dimen is None:
            dimen = (
                len(self._items[0])
                if self._items and isinstance(self._items[0], tuple)
                else 1
            )
        self.dimen = dimen

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item: Any) -> bool:
        return item in self.pos

    def __getitem__(self, position: int) -> Any:
        return self._items[position]

    def index(self, item: Any) -> int:
        return self.pos[item]

    def indices(self, items: Iterable[Any]) -> np.ndarray:
        return np.fromiter((self.pos[item] for item in items), dtype=int)

    def __repr__(self) -> str:
        return f"IndexSet({self._items!r})"


def index_product(*sets: Iterable[Any]) -> list[tuple]:
    """Cartesian product of sets with tuple indices flattened."""
    return [
        tuple(part for item in combo for part in _flatten(item))
        for combo in product(*sets)
    ]


class ParamDict(dict):
    """Dictionary of parameter values with an optional default for missing keys."""

    def __init__(self, data: dict | None = None, default: Any = _NO_DEFAULT) -> None:
        super().__init__(data or {})
        self.default = default

    def __missing__(self, key: Any) -> Any:
        if self.default is _NO_DEFAULT:
            raise KeyError(key)
        return self.default


def sparse(
    rows: Iterable[int], cols: Iterable[int], vals: Iterable[float], shape: tuple
) -> sp.csr_matrix:
    """Build a CSR matrix, summing duplicate entries."""
    return sp.coo_matrix(
        (
            np.asarray(list(vals), dtype=float),
            (np.asarray(list(rows), dtype=int), np.asarray(list(cols), dtype=int)),
        ),
        shape=shape,
    ).tocsr()


@dataclass
class VarSpec:
    """Index information needed to turn a variable array back into a table."""

    name: str
    variable: cp.Variable
    axes: tuple
    time: bool
    boolean: bool = False


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

    def __len__(self) -> int:
        return len(self.rows)

    def constraints(self) -> list:
        return list(self.pieces)


class CvxModel:
    """Attribute container for sets, parameters, variables, and constraints."""

    def __init__(self) -> None:
        object.__setattr__(self, "_constraint_names", [])
        object.__setattr__(self, "_var_names", [])
        object.__setattr__(self, "var_specs", {})
        object.__setattr__(self, "objective_expression", None)
        object.__setattr__(self, "problem", None)

    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any:
            """Components are attached dynamically; declare that to type checkers."""
            ...

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

    def add_var(
        self,
        name: str,
        *axes: Any,
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
        if name not in self._var_names:
            self._var_names.append(name)
        self.var_specs[name] = VarSpec(name, variable, tuple(axes), time, boolean)
        object.__setattr__(self, name, variable)
        return variable

    def set_values(self, name: str, values: Any) -> None:
        """Warm start variable ``name`` from an array broadcastable to its shape."""
        variable = self.var_specs[name].variable
        variable.value = np.broadcast_to(
            np.asarray(values, dtype=float), variable.shape
        ).copy()

    def create_constraint(self, name: str) -> ConstraintRecord:
        if name in self._constraint_names:
            raise ValueError(f"Constraint {name!r} already exists on this model")
        record = ConstraintRecord(name)
        object.__setattr__(self, name, record)
        self._constraint_names.append(name)
        return record

    def del_component(self, name: str) -> None:
        if name in self._constraint_names:
            self._constraint_names.remove(name)
        if name in self._var_names:
            self._var_names.remove(name)
            self.var_specs.pop(name, None)
        if name in self.__dict__:
            object.__delattr__(self, name)

    @property
    def constraint_names(self) -> list[str]:
        return list(self._constraint_names)

    @property
    def var_names(self) -> list[str]:
        return list(self._var_names)

    def all_constraints(self) -> list:
        collected = []
        for name in self._constraint_names:
            collected.extend(getattr(self, name).pieces)
        return collected


def add_constraint(
    model: CvxModel, name: str, constraint: Any, rows: Iterable[Any] = ()
) -> ConstraintRecord:
    """Register ``constraint`` (one vectorized constraint or a list) as ``name``.

    ``rows`` are the row keys the constraint covers, in row order, without time.
    """
    record = model.create_constraint(name)
    pieces = constraint if isinstance(constraint, (list, tuple)) else [constraint]
    for piece in pieces:
        if isinstance(piece, bool):
            if piece:
                continue
            raise ValueError(f"Constraint {name!r} is trivially infeasible")
        record.pieces.append(piece)
    record.rows = list(rows)
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


def column(values: Any) -> np.ndarray:
    """Reshape a per-row vector to ``(n, 1)`` so it broadcasts over time."""
    return np.asarray(values, dtype=float).reshape(-1, 1)
