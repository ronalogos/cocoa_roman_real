from __future__ import absolute_import, division, print_function

import numpy as np

from cobaya.log import LoggedError

from cobaya.likelihoods.roman_real._cosmolike_prototype_base import _cosmolike_prototype_base


class cosmic_shear_2pt_map3(_cosmolike_prototype_base):
    """Joint cosmic shear 2pt + map3 likelihood.

    The 2pt (xi) part is computed through the existing CosmoLike interface.
    The map3 part is expected from an external theory provider (e.g. a custom
    theory code), and the full joint covariance is read from user-provided files.
    """

    def initialize(self):
        if self.use_emulator:
            raise LoggedError(
                self.log,
                "cosmic_shear_2pt_map3 does not support use_emulator=True. "
                "Use the non-emulator pipeline for xi + map3.",
            )

        super(cosmic_shear_2pt_map3, self).initialize(probe="xi")

        # Joint (xi + map3) files, independent from the CosmoLike internal files.
        self.joint_data_vector_file = getattr(self, "joint_data_vector_file", None)
        self.joint_cov_file = getattr(self, "joint_cov_file", None)
        self.joint_mask_file = getattr(self, "joint_mask_file", None)

        if self.joint_data_vector_file is None or self.joint_cov_file is None:
            raise LoggedError(
                self.log,
                "joint_data_vector_file and joint_cov_file must be provided for "
                "cosmic_shear_2pt_map3.",
            )

        self.joint_data_vector = np.loadtxt(self.joint_data_vector_file, dtype="float64")
        self.joint_cov = np.loadtxt(self.joint_cov_file, dtype="float64")

        if self.joint_cov.ndim != 2 or self.joint_cov.shape[0] != self.joint_cov.shape[1]:
            raise LoggedError(self.log, "joint_cov_file must contain a square matrix.")

        if self.joint_data_vector.ndim != 1:
            self.joint_data_vector = np.ravel(self.joint_data_vector)

        if self.joint_cov.shape[0] != self.joint_data_vector.size:
            raise LoggedError(
                self.log,
                "Joint covariance size (%d) does not match joint data vector size (%d).",
                self.joint_cov.shape[0],
                self.joint_data_vector.size,
            )

        if self.joint_mask_file is None:
            self.joint_mask = np.ones(self.joint_data_vector.size, dtype=bool)
        else:
            tmp_mask = np.loadtxt(self.joint_mask_file)
            tmp_mask = np.ravel(tmp_mask).astype(int)
            if tmp_mask.size != self.joint_data_vector.size:
                raise LoggedError(
                    self.log,
                    "Joint mask size (%d) does not match joint data vector size (%d).",
                    tmp_mask.size,
                    self.joint_data_vector.size,
                )
            self.joint_mask = tmp_mask != 0

        self.inv_joint_cov = np.linalg.inv(self.joint_cov[np.ix_(self.joint_mask, self.joint_mask)])

    def get_requirements(self):
        reqs = super(cosmic_shear_2pt_map3, self).get_requirements()
        reqs["map3"] = None
        return reqs

    def _get_map3_from_provider(self):
        if hasattr(self.provider, "get_map3"):
            map3 = self.provider.get_map3()
        elif hasattr(self.provider, "get_mass_aperture"):
            map3 = self.provider.get_mass_aperture()
        else:
            raise LoggedError(
                self.log,
                "Provider must implement get_map3() or get_mass_aperture() for "
                "cosmic_shear_2pt_map3.",
            )
        return np.asarray(map3, dtype="float64").ravel()

    def get_datavector(self, **params):
        xi_dv = np.asarray(
            super(cosmic_shear_2pt_map3, self).get_datavector(**params),
            dtype="float64",
        ).ravel()
        map3_dv = self._get_map3_from_provider()
        joint_theory = np.concatenate((xi_dv, map3_dv))

        if joint_theory.size != self.joint_data_vector.size:
            raise LoggedError(
                self.log,
                "Joint theory vector size (%d) does not match joint data vector size (%d). "
                "Check xi/map3 ordering and lengths.",
                joint_theory.size,
                self.joint_data_vector.size,
            )

        return joint_theory

    def compute_logp(self, datavector):
        residual = np.asarray(datavector, dtype="float64") - self.joint_data_vector
        masked_residual = residual[self.joint_mask]
        chi2 = float(masked_residual @ self.inv_joint_cov @ masked_residual)
        return -0.5 * chi2

    def logp(self, **params):
        return self.compute_logp(self.get_datavector(**params))
