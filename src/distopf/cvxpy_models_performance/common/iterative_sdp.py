"""Iterative rank penalty for the SDP relaxation (experimental).

Each branch PSD matrix M is the real embedding [[A, -B], [B, A]] of a Hermitian
matrix, so its eigenvalues come in equal pairs. Rank-1 Hermitian means the top
pair carries all of the trace. Each iteration adds the linear penalty
``mu * (Tr(M) - u'Mu - w'Mw)`` with ``u`` the previous top eigenvector and
``w = J u`` its pair, then re-solves the SDP.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import cvxpy as cp
import numpy as np

from distopf.cvxpy_models_performance.common.model import CvxModel
from distopf.cvxpy_models_performance.common.objectives import set_objective
from distopf.cvxpy_models_performance.common.results import CvxResult
from distopf.cvxpy_models_performance.common.solvers import build_problem, solve


def psd_matrices(model: CvxModel) -> list:
    """Branch PSD matrix expressions registered by the SDP relaxation."""
    return [piece.args[0] for piece in model.psd_constraint.pieces]


def rank_gaps(matrices: list) -> tuple[list, np.ndarray]:
    """Dominant eigenvectors and per-matrix (Tr - lambda_1 - lambda_2) / Tr."""
    vectors, gaps = [], []
    for matrix in matrices:
        value = np.asarray(matrix.value)
        eigvals, eigvecs = np.linalg.eigh((value + value.T) / 2)
        vectors.append(eigvecs[:, -1])
        gaps.append((eigvals.sum() - eigvals[-2:].sum()) / max(eigvals.sum(), 1e-12))
    return vectors, np.array(gaps)


def complex_partner(u: np.ndarray) -> np.ndarray:
    """J u with J = [[0, -I], [I, 0]]: the paired eigenvector of the real embedding."""
    half = len(u) // 2
    return np.concatenate([-u[half:], u[:half]])


def matrix_entry_maps(matrices: list, variables: list) -> list:
    """Per matrix, entries as (variable index, flat positions, signs, rows, cols).

    Temporarily sets variable values to unique ids so each entry reveals its source.
    """
    solution = [v.value for v in variables]
    offsets = np.cumsum([0] + [v.size for v in variables])
    for v, offset in zip(variables, offsets):
        v.value = (offset + 1 + np.arange(v.size)).reshape(v.shape).astype(float)
    maps = []
    for matrix in matrices:
        ids = np.asarray(matrix.value)
        rows, cols = np.nonzero(ids)
        signed = ids[rows, cols]
        flat_ids = np.abs(signed).astype(int) - 1
        # Entries that are sums or offsets of variables would decode to bad ids
        if (
            not np.array_equal(np.abs(signed), flat_ids + 1)
            or flat_ids.max() >= offsets[-1]
        ):
            raise ValueError(
                "PSD matrix entries must each be +/- a single variable entry."
            )
        var_idx = np.searchsorted(offsets, flat_ids, side="right") - 1
        entries = []
        for i in np.unique(var_idx):
            sel = var_idx == i
            entries.append(
                (
                    i,
                    flat_ids[sel] - offsets[i],
                    np.sign(signed[sel]),
                    rows[sel],
                    cols[sel],
                )
            )
        maps.append(entries)
    for v, value in zip(variables, solution):
        v.value = value
    return maps


def solve_iterative_sdp(
    model: CvxModel,
    solver: str = "CLARABEL",
    mu: float = 1e-2,
    growth: float = 2.0,
    mu_max: float = 10.0,
    rho: float = 0.0,
    max_iter: int = 200,
    tol: float = 1e-2,
    patience: int = 20,
    verbose: bool = False,
    update_angles: Callable[[], None] | None = None,
) -> tuple[CvxResult, list[tuple[float, float]]]:
    """Solve an SDP model, then tighten rank with the iterative penalty.

    ``model`` must be built with ``sdp_relaxation=True`` and have its objective set.
    ``mu`` grows by ``growth`` (up to ``mu_max``) whenever the max rank gap falls by
    less than 10%. ``rho > 0`` adds a proximal term to the previous iterate. Stops
    at ``tol``, or when the gap improves under 1% over ``patience`` iterations.
    ``update_angles`` runs before each solve, while the model holds the previous
    iterate, to refresh the constant current-angle parameters.
    Returns the result and a history of ``(mu, max rank gap)``.
    """
    solve(model, solver=solver, duals=False)  # k=0: plain SDP
    base_objective = model.objective_expression
    matrices = psd_matrices(model)
    vectors, gaps = rank_gaps(matrices)
    history = [(0.0, float(gaps.max()))]

    variables = model.problem.variables()
    if rho > 0:
        anchors = [cp.Parameter(v.shape, value=v.value) for v in variables]
    entry_maps = matrix_entry_maps(matrices, variables)
    coefs = [cp.Parameter(v.shape) for v in variables]
    objective = base_objective + cp.sum(
        [cp.sum(cp.multiply(c, v)) for c, v in zip(coefs, variables)]
    )
    if rho > 0:
        objective += rho * cp.sum(
            [cp.sum_squares(v - a) for v, a in zip(variables, anchors)]
        )
    set_objective(model, objective)
    problem = build_problem(model)

    for k in range(1, max_iter + 1):
        if gaps.max() < tol:
            break
        flat = [np.zeros(v.size) for v in variables]
        for entries, u in zip(entry_maps, vectors):
            w = complex_partner(u)
            weight = mu * (np.eye(len(u)) - np.outer(u, u) - np.outer(w, w))
            for var_idx, positions, signs, rows, cols in entries:
                np.add.at(flat[var_idx], positions, signs * weight[rows, cols])
        for c, v, f in zip(coefs, variables, flat):
            c.value = f.reshape(v.shape)
        if update_angles is not None:
            update_angles()
        problem.solve(solver=solver)
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Iteration {k} failed: {problem.status}")
        if rho > 0:
            for a, v in zip(anchors, variables):
                a.value = v.value
        previous_gap = gaps.max()
        vectors, gaps = rank_gaps(matrices)
        history.append((mu, float(gaps.max())))
        if verbose:
            print(f"k={k} mu={mu:.1e} max rank gap {gaps.max():.3e}")
        if gaps.max() > 0.9 * previous_gap:
            mu = min(mu * growth, mu_max)
        if len(history) > patience:
            old_gap = history[-patience - 1][1]
            if old_gap - gaps.max() < 0.01 * old_gap:
                break

    if gaps.max() >= tol:
        warnings.warn(
            f"Iterative SDP stopped with max rank gap {gaps.max():.3g} >= tol {tol}; "
            "the solution may not be physically feasible."
        )
    set_objective(model, base_objective)
    result = CvxResult(
        model,
        problem,
        solve_time=problem.solver_stats.solve_time or 0.0,
        log="",
        objective_value=float(base_objective.value),
        extract_duals=False,
    )
    return result, history
