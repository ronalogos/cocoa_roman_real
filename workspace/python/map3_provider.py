from __future__ import annotations

import numpy as np
from cobaya.theory import Theory


class map3_provider(Theory):
    """Minimal Cobaya theory component that serves a map3 vector from file.

    This is a bridge module for integrating external map3 pipelines.
    Replace/extend this class with the direct 3pcf_integrator computation when
    available in the runtime environment.
    """

    map3_file: str | None = None

    def initialize(self):
        if self.map3_file is None:
            raise ValueError("map3_provider requires 'map3_file' in the theory block")
        self._map3 = np.loadtxt(self.map3_file, dtype="float64").ravel()

    def get_requirements(self):
        return {}

    def must_provide(self, **requirements):
        return {}

    def calculate(self, state, want_derived=True, **params_values_dict):
        state["map3"] = self._map3

    def get_map3(self):
        return self.current_state["map3"]
