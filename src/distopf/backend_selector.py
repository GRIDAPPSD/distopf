"""Backend selection and routing logic for Case.run_opf().

This module provides a single source of truth for backend selection rules
and delegates run_opf operations to the appropriate backend implementation.
"""

from typing import Optional, Dict, Type
from distopf.backends import MatrixBackend, MultiperiodBackend, PyomoBackend, Backend


class BackendSelector:
    """Select and route to the appropriate optimization backend."""

    # Backend factory mapping: backend_name -> Backend class
    BACKEND_FACTORY: Dict[str, Type[Backend]] = {
        "matrix": MatrixBackend,
        "multiperiod": MultiperiodBackend,
        "pyomo": PyomoBackend,
    }

    # Backend selection rules: (condition_function, backend_name)
    # Rules are evaluated in order; first match wins
    AUTO_SELECT_RULES = [
        (lambda case: case.n_steps > 1, "multiperiod"),
        (
            lambda case: case.bat_data is not None and len(case.bat_data) > 0,
            "multiperiod",
        ),
        (
            lambda case: case.schedules is not None and len(case.schedules) > 0,
            "multiperiod",
        ),
    ]

    SUPPORTED_BACKENDS = set(BACKEND_FACTORY.keys())

    def __init__(self, case):
        """Initialize selector with a Case instance."""
        self.case = case
        self.backend_override = None

    def select(self) -> str:
        """Auto-select the best backend based on case properties.

        Uses AUTO_SELECT_RULES to determine backend. Can be overridden
        by calling select(backend="explicit_backend").

        Returns
        -------
        str
            Selected backend name ("matrix", "multiperiod", or "pyomo")
        """
        if self.backend_override:
            return self.backend_override

        # Apply auto-selection rules
        for rule_func, backend in self.AUTO_SELECT_RULES:
            if rule_func(self.case):
                return backend

        # Default to single-period matrix model
        return "matrix"

    def validate_backend(self, backend: Optional[str]) -> str:
        """
        Validate and normalize backend name.

        Parameters
        ----------
        backend : str or None
            Backend name to validate

        Returns
        -------
        str
            Validated backend name (lowercase)

        Raises
        ------
        ValueError
            If backend is not recognized
        """
        if backend is None:
            return self.select()

        backend = backend.lower().strip()

        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown backend: '{backend}'. "
                f"Supported backends: {', '.join(sorted(self.SUPPORTED_BACKENDS))}"
            )

        return backend

    def route_opf(
        self,
        objective,
        control_variable=None,
        control_regulators=False,
        control_capacitors=False,
        backend=None,
        raw_result=False,
        return_result_object=True,
        **kwargs,
    ):
        """Route run_opf to the appropriate backend implementation.

        Parameters
        ----------
        objective : str or Callable
            Objective function for optimization
        control_variable : str, optional
            Control variable for generators
        control_regulators : bool
            Enable regulator tap optimization
        control_capacitors : bool
            Enable capacitor switching optimization
        backend : str, optional
            Override backend selection
        raw_result : bool
            Return raw backend result if True
        return_result_object : bool, default True
            If True, return OpfResult; if False, return tuple (backward compat)
        **kwargs
            Additional backend-specific arguments

        Returns
        -------
        OpfResult or tuple
            OpfResult if return_result_object=True, else (voltages, power_flows, p_gens, q_gens)
        """
        from distopf.result import OpfResult

        # Validate and select backend
        if backend is None:
            backend = self.select()
        else:
            backend = self.validate_backend(backend)

        # Instantiate backend using factory
        backend_class = self.BACKEND_FACTORY[backend]
        backend_obj = backend_class(self.case)

        # Solve using backend
        result_tuple = backend_obj.solve(
            objective=objective,
            control_variable=control_variable,
            control_regulators=control_regulators,
            control_capacitors=control_capacitors,
            raw_result=raw_result,
            **kwargs,
        )

        # If raw_result requested, return as-is
        if raw_result:
            return result_tuple

        # Unpack tuple and create OpfResult
        if return_result_object:
            voltages, power_flows, p_gens, q_gens = result_tuple
            return OpfResult(
                voltages=voltages,
                power_flows=power_flows,
                p_gens=p_gens,
                q_gens=q_gens,
                case=self.case,
                model=backend_obj.model,
            )
        else:
            # Backward compatibility: return tuple
            return result_tuple


__all__ = ["BackendSelector"]
