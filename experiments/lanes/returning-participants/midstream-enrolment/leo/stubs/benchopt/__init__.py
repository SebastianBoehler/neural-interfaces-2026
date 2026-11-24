"""Stub of benchopt for the Leonardo venv (no benchopt installed, no internet on compute nodes): CompetSolver only needs a base class to subclass;
the NI-V worker calls Solver().load_model(...) and the package's own methods, never benchopt's runner."""


class BaseSolver:
    parameters = {}

    def __init__(self, **kwargs):
        pass
