"""Light model container mirroring the Pyomo ConcreteModel idioms used by DistOPF.

CVXPY has no Set, indexed Param, or rule-based Constraint. This module supplies
small stand-ins so that the model-building code can keep the same structure as
``distopf.pyomo_models``:

- ``IndexSet``: ordered set with fast membership and a ``dimen`` attribute.
- ``ParamDict``: dict of numeric data with an optional default for missing keys.
- ``VarDict``: dict of scalar ``cp.Variable`` objects keyed by index tuples.
- ``ConstraintDict``: dict of scalar CVXPY constraints keyed by index tuples.
- ``CvxModel``: attribute container that tracks variables and constraints.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import cvxpy as cp
import numpy as np

_NO_DEFAULT = object()


def _flatten(item: Any) -> tuple:
    return item if isinstance(item, tuple) else (item,)


class IndexSet:
    """Ordered collection of unique indices with O(1) membership tests."""

    def __init__(self, items: Iterable[Any] = (), dimen: int | None = None) -> None:
        self._items = list(dict.fromkeys(items))
        self._lookup = set(self._items)
        if dimen is None:
            dimen = len(self._items[0]) if self._items and isinstance(
                self._items[0], tuple
            ) else 1
        self.dimen = dimen

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item: Any) -> bool:
        return item in self._lookup

    def __getitem__(self, position: int) -> Any:
        return self._items[position]

    def __mul__(self, other: IndexSet) -> list[tuple]:
        return [_flatten(a) + _flatten(b) for a, b in product(self, other)]

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


class VarDict(dict):
    """Dictionary of scalar ``cp.Variable`` objects keyed by index tuples."""

    def __init__(
        self,
        keys: Iterable[Any],
        nonneg: bool = False,
        boolean: bool = False,
        initialize: Any = None,
    ) -> None:
        super().__init__()
        for key in keys:
            variable = cp.Variable(nonneg=nonneg, boolean=boolean)
            if initialize is not None:
                value = initialize(*_flatten(key)) if callable(initialize) else initialize
                variable.value = np.float64(value)
            self[key] = variable
        self.name = ""
        self.nonneg = nonneg
        self.boolean = boolean
        self.model_ref: CvxModel | None = None

    @property
    def local_name(self) -> str:
        return self.name

    def dim(self) -> int:
        for key in self:
            return len(_flatten(key))
        return 0

    def model(self) -> CvxModel | None:
        return self.model_ref

    def extract_values(self) -> dict[Any, float]:
        values = {}
        for key, variable in self.items():
            value = variable.value
            values[key] = float("nan") if value is None else float(value)
        return values

    def set_values(self, values: dict[Any, float]) -> None:
        for key, value in values.items():
            self[key].value = np.float64(value)


class ConstraintDict(dict):
    """Dictionary mapping an index to the list of CVXPY constraints it produced.

    Equalities and one-sided inequalities store one constraint. Ranged
    inequalities store two: ``[lower, upper]``.
    """

    def __init__(self, name: str = "") -> None:
        super().__init__()
        self.name = name

    @property
    def local_name(self) -> str:
        return self.name

    def add(self, index: Any, constraint: Any) -> None:
        if isinstance(constraint, bool):
            if constraint:
                return
            raise ValueError(f"Constraint {self.name}[{index}] is trivially infeasible")
        self[index] = [constraint]

    def add_range(self, index: Any, lower: Any, expression: Any, upper: Any) -> None:
        pieces = []
        if lower is not None:
            pieces.append(expression >= lower)
        if upper is not None:
            pieces.append(expression <= upper)
        self[index] = pieces

    def constraints(self) -> list:
        return [piece for pieces in self.values() for piece in pieces]


class CvxModel:
    """Attribute container for sets, params, variables, and constraints."""

    def __init__(self) -> None:
        object.__setattr__(self, "_constraint_names", [])
        object.__setattr__(self, "_var_names", [])
        object.__setattr__(self, "objective_expression", None)
        object.__setattr__(self, "problem", None)

    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any:
            """Components are attached dynamically; declare that to type checkers."""
            ...

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, VarDict):
            value.name = name
            value.model_ref = self
            if name not in self._var_names:
                self._var_names.append(name)
        object.__setattr__(self, name, value)

    def create_constraint(self, name: str) -> ConstraintDict:
        if name in self._constraint_names:
            raise ValueError(f"Constraint {name!r} already exists on this model")
        container = ConstraintDict(name)
        object.__setattr__(self, name, container)
        self._constraint_names.append(name)
        return container

    def del_component(self, name: str) -> None:
        if name in self._constraint_names:
            self._constraint_names.remove(name)
        if name in self._var_names:
            self._var_names.remove(name)
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
            collected.extend(getattr(self, name).constraints())
        return collected


def esum(terms: Iterable[Any]) -> Any:
    """Sum terms (0 when empty)."""
    return sum(terms)


class _Skip:
    def __repr__(self) -> str:
        return "SKIP"


SKIP = _Skip()


def add_constraint(model: CvxModel, name: str, index_sets: Iterable[Any], rule) -> ConstraintDict:
    """Create ``model.<name>`` by evaluating ``rule(model, *index)`` for every index.

    This mirrors ``pyo.Constraint(*sets, rule=rule)``. The rule may return:

    - a CVXPY constraint (``==``, ``<=``, ``>=``),
    - ``SKIP`` (or ``None``) to omit that index,
    - ``(lower, expression, upper)`` for a ranged constraint; either bound may
      be ``None``.
    """
    container = model.create_constraint(name)
    keys = index_product(*index_sets) if index_sets else [()]
    for key in keys:
        args = key
        if len(key) == 1:
            key = key[0]
        result = rule(model, *args)
        if result is None or result is SKIP:
            continue
        if isinstance(result, tuple):
            lower, expression, upper = result
            container.add_range(key, lower, expression, upper)
        else:
            container.add(key, result)
    return container
