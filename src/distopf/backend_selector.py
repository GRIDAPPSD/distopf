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

    SUPPORTED_BACKENDS = set(BACKEND_FACTORY.keys())

    def __init__(self, case):
        """Initialize selector with a Case instance."""
        self.case = case
        self.backend_override = None

    def select(self, control_regulators=False, control_capacitors=False) -> str:
        """Auto-select the best backend based on case properties.

        Returns
        -------
        str
            Selected backend name ("matrix", "multiperiod", or "pyomo")
        """
        if self.backend_override:
            return self.backend_override

        if control_regulators or control_capacitors:
            return "matrix"
        return "multiperiod"

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

    def solve(
        self,
        objective,
        control_variable=None,
        control_regulators=False,
        control_capacitors=False,
        backend=None,
        raw_result=False,
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
        **kwargs
            Additional backend-specific arguments

        Returns
        -------
        PowerFlowResult
            Unified result object with all OPF outputs
        """
        # Validate and select backend
        if backend is None:
            backend = self.select(
                control_capacitors=control_capacitors,
                control_regulators=control_regulators,
            )
        else:
            backend = self.validate_backend(backend)

        # Instantiate backend using factory
        backend_class = self.BACKEND_FACTORY[backend]
        backend_obj = backend_class(self.case)

        # Set control variable if specified (updates gen_data)
        if control_variable is not None:
            backend_obj.set_control_variable(control_variable)

        # Solve using backend - returns PowerFlowResult directly
        return backend_obj.solve(
            objective=objective,
            control_regulators=control_regulators,
            control_capacitors=control_capacitors,
            raw_result=raw_result,
            **kwargs,
        )


__all__ = ["BackendSelector"]
