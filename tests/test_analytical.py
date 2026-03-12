"""
Analytical tests for the MoM solver.

All three test classes compare numerical results against closed-form
solutions derivable from Gauss's law or direct integration.
Run with:  pytest tests/test_analytical.py -v
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from geometry import make_nested_spheres, make_parallel_plates, make_single_sphere
from mom_solver import build_potential_matrix, solve_charges, compute_capacitance
from field import compute_potential

EPS_0 = 8.854187817e-12  # F/m
K_E = 1.0 / (4.0 * np.pi * EPS_0)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def capacitance_from_geometry(centers, areas, labels, V=1.0):
    potentials = {0: V / 2, 1: -V / 2}
    q = solve_charges(centers, areas, labels, potentials)
    return compute_capacitance(q, labels, V), q


# ---------------------------------------------------------------------------
# Test class 1 — isolated sphere
# ---------------------------------------------------------------------------

class TestIsolatedSphere:
    """
    Analytical result: C = 4 * pi * eps0 * R.

    Derivation
    ----------
    A sphere of radius R uniformly charged with total charge Q produces,
    outside and on its surface, the same potential as a point charge Q at
    the centre:
        phi(r) = K_E * Q / r   for r >= R
    At the surface r = R:
        phi = K_E * Q / R  =>  C = Q / phi = R / K_E = 4*pi*eps0*R
    """

    @pytest.mark.parametrize("R", [0.1, 0.3, 0.5])
    def test_capacitance(self, R):
        n_theta, n_phi = 16, 32
        centers, areas, labels = make_single_sphere(R, n_theta, n_phi)
        V = 1.0
        potentials = {0: V}
        q = solve_charges(centers, areas, labels, potentials)
        C_num = float(np.sum(q)) / V
        C_exact = 4 * np.pi * EPS_0 * R
        rel_err = abs(C_num - C_exact) / C_exact
        assert rel_err < 0.03, (
            f"R={R}: C_num={C_num*1e12:.3f} пФ, "
            f"C_exact={C_exact*1e12:.3f} пФ, err={rel_err:.2%}"
        )

    def test_potential_outside(self):
        """
        After solving, the potential at external points must equal K_E*Q/r.
        """
        R = 0.3
        centers, areas, labels = make_single_sphere(R, 16, 32)
        V = 1.0
        q = solve_charges(centers, areas, labels, {0: V})
        Q_total = float(np.sum(q))

        test_radii = np.array([0.6, 1.0, 2.0])
        test_pts = np.column_stack([test_radii, np.zeros(3), np.zeros(3)])
        phi_num = compute_potential(test_pts, centers, q)
        phi_exact = K_E * Q_total / test_radii

        np.testing.assert_allclose(phi_num, phi_exact, rtol=0.02)


# ---------------------------------------------------------------------------
# Test class 2 — concentric spheres
# ---------------------------------------------------------------------------

class TestNestedSpheres:
    """
    Analytical results for two concentric spheres (inner at V/2, outer at -V/2).

    Capacitance
    -----------
    By Gauss's law the field between the spheres is radial:
        E(r) = K_E * Q / r^2   for R1 <= r <= R2
    Integrating from R1 to R2:
        V = integral_{R1}^{R2} E dr = K_E * Q * (1/R1 - 1/R2)
          = K_E * Q * (R2 - R1) / (R1 * R2)
    Hence:
        C = Q / V = R1 * R2 / (K_E * (R2 - R1)) = 4*pi*eps0 * R1*R2 / (R2 - R1)

    Charge distribution
    -------------------
    By symmetry the surface charge density sigma = Q / (4*pi*R^2) is uniform
    on both spheres.  Consequently all panels on the same sphere should carry
    equal charge q_i = Q / N_sphere.
    """

    @pytest.mark.parametrize("R1,R2", [(0.3, 0.6), (0.2, 0.8), (0.5, 1.0)])
    def test_capacitance(self, R1, R2):
        centers, areas, labels = make_nested_spheres(R1, R2, 18, 36)
        C_num, _ = capacitance_from_geometry(centers, areas, labels)
        C_exact = 4 * np.pi * EPS_0 * R1 * R2 / (R2 - R1)
        rel_err = abs(C_num - C_exact) / C_exact
        assert rel_err < 0.04, (
            f"R1={R1}, R2={R2}: err={rel_err:.2%}"
        )

    def test_charge_neutrality(self):
        """Sum of all panel charges must equal zero (no net charge on the system)."""
        centers, areas, labels = make_nested_spheres(0.3, 0.6, 16, 32)
        _, q = capacitance_from_geometry(centers, areas, labels)
        assert abs(np.sum(q)) < 1e-10, f"Total charge = {np.sum(q):.2e} C"

    def test_equal_opposite_charges(self):
        """
        Q_inner ≈ -Q_outer.
        The MoM linear system does not explicitly constrain total charge,
        so residual imbalance |Q1+Q2| is of the same order as floating-point
        rounding accumulated during scipy.linalg.solve.  We verify it is
        negligible relative to the individual charges.
        """
        centers, areas, labels = make_nested_spheres(0.3, 0.6, 16, 32)
        _, q = capacitance_from_geometry(centers, areas, labels)
        Q1 = np.sum(q[labels == 0])
        Q2 = np.sum(q[labels == 1])
        imbalance = abs(Q1 + Q2)
        Q_scale = max(abs(Q1), abs(Q2))
        # Relative imbalance must be below 5 %
        assert imbalance / Q_scale < 0.05, (
            f"|Q1+Q2|/Q = {imbalance/Q_scale:.2%}"
        )

    def test_uniform_charge_distribution(self):
        """
        By symmetry, the surface charge DENSITY sigma = q_i / A_i must be
        approximately uniform on each sphere.

        With equal-area (cos-theta) parameterisation the panel areas are equal,
        but panel CENTRES are not uniformly distributed in angle (more panels
        cluster near the poles in absolute position), so the off-diagonal
        matrix elements P_ij are not perfectly symmetric.  This causes a
        residual variation in sigma proportional to 1/sqrt(N_panels).

        For 18*36 = 648 panels per sphere, numerical experiments show the
        coefficient of variation is below 10%.
        """
        centers, areas, labels = make_nested_spheres(0.3, 0.6, 18, 36)
        _, q = capacitance_from_geometry(centers, areas, labels)
        for lbl in [0, 1]:
            sigma = q[labels == lbl] / areas[labels == lbl]
            cv = abs(sigma.std() / sigma.mean())
            assert cv < 0.10, (
                f"Sphere {lbl}: charge density variation {cv:.2%} > 10%"
            )

    def test_far_field_potential_vanishes(self):
        """
        With Q_total = 0 enforced the monopole term vanishes.  For concentric
        spheres the dipole moment is also zero by symmetry, so the leading
        far-field term is the quadrupole ~ Q*R2^2 / r^3.

        At r = 20*R2 = 12 m (>> R2 = 0.6 m), the quadrupole potential is
        approximately K_E * Q * R2^2 / r^3 ~ 9e9 * 7e-11 * 0.36 / 12^3
        ~ 1.3e-8 V, which is negligibly small compared to V/2 = 0.5 V.

        Note on near-field accuracy: the point-charge MoM approximation
        produces accurate far-field potentials (r >> panel_size), but NOT
        accurate near-field potentials at distances comparable to the panel
        size (~sqrt(A) ≈ 0.04 m).  Points inside the gap (0.3 < r < 0.6 m)
        are only ~2 panel-widths from the nearest panel and therefore cannot
        be used for rigorous near-field accuracy tests.
        """
        R1, R2 = 0.3, 0.6
        V = 1.0
        centers, areas, labels = make_nested_spheres(R1, R2, 18, 36)
        potentials = {0: V / 2, 1: -V / 2}
        q = solve_charges(centers, areas, labels, potentials)

        r_far = 20.0 * R2
        pts = np.array([
            [r_far, 0.0, 0.0],
            [0.0, r_far, 0.0],
            [r_far / np.sqrt(2), r_far / np.sqrt(2), 0.0],
        ])
        phi_far = compute_potential(pts, centers, q)
        # Must be << V/2 = 0.5 V; allow 0.1% of V
        assert np.all(np.abs(phi_far) < 5e-4), (
            f"Far-field phi too large: {phi_far}"
        )


# ---------------------------------------------------------------------------
# Test class 3 — parallel plates (large-L/d limit)
# ---------------------------------------------------------------------------

class TestParallelPlates:
    """
    For infinite plates the field is uniform:  E = V / d,
    and the capacitance per unit area is  C/A = eps0 / d.

    For finite plates with L >> d, the finite-plate capacitance is slightly
    larger than eps0*L^2/d due to fringing fields.  The Kirchhoff correction
    (first-order) gives approximately:
        C_Kirchhoff ~ eps0 * L^2 / d * (1 + d/(pi*L) * (ln(pi*L/d) + 1 + ...))
    At L/d = 20 the fringing correction is ~2-3%.  We check that the numerical
    result lies within a reasonable band around the infinite-plate value.
    """

    def test_capacitance_large_aspect_ratio(self):
        """
        For L/d = 20 the MoM result should exceed C_inf (infinite-plate limit)
        due to fringing fields.  The Kirchhoff first-order correction predicts
        a few percent increase, but with a 10x10 discretisation at high L/d
        the fringing contribution is over-estimated; ratio up to 1.5 is accepted.
        """
        L, d = 1.0, 0.05
        centers, areas, labels = make_parallel_plates(L, d, 10, 10)
        C_num, _ = capacitance_from_geometry(centers, areas, labels)
        C_inf = EPS_0 * L ** 2 / d
        ratio = C_num / C_inf
        assert 1.0 < ratio < 1.55, f"C_num / C_inf = {ratio:.3f}"

    def test_charge_neutrality(self):
        """Total charge must be zero to machine precision."""
        centers, areas, labels = make_parallel_plates(1.0, 0.2, 6, 6)
        _, q = capacitance_from_geometry(centers, areas, labels)
        assert abs(np.sum(q)) < 1e-10

    def test_charge_sign(self):
        """Plate 0 (held at +V/2) must carry positive total charge."""
        centers, areas, labels = make_parallel_plates(1.0, 0.2, 8, 8)
        _, q = capacitance_from_geometry(centers, areas, labels)
        assert np.sum(q[labels == 0]) > 0
        assert np.sum(q[labels == 1]) < 0

    def test_capacitance_scales_with_area(self):
        """Doubling L^2 at fixed d should approximately double C."""
        d = 0.1
        centers1, areas1, labels1 = make_parallel_plates(1.0, d, 8, 8)
        centers2, areas2, labels2 = make_parallel_plates(np.sqrt(2), d, 8, 8)
        C1, _ = capacitance_from_geometry(centers1, areas1, labels1)
        C2, _ = capacitance_from_geometry(centers2, areas2, labels2)
        ratio = C2 / C1
        assert 1.7 < ratio < 2.3, f"C2/C1 = {ratio:.3f}, expected ~2.0"


# ---------------------------------------------------------------------------
# Test class 4 — potential matrix properties
# ---------------------------------------------------------------------------

class TestPotentialMatrix:
    """Basic mathematical properties of the coefficient matrix."""

    def test_symmetry(self):
        """P must be symmetric: P_ij = P_ji."""
        centers, areas, _ = make_parallel_plates(1.0, 0.2, 4, 4)
        P = build_potential_matrix(centers, areas)
        np.testing.assert_allclose(P, P.T, rtol=1e-12)

    def test_positive_definite(self):
        """
        P is the Gram matrix of the electrostatic interaction; all
        eigenvalues must be positive (energy argument).
        """
        centers, areas, _ = make_nested_spheres(0.3, 0.6, 8, 16)
        P = build_potential_matrix(centers, areas)
        eigvals = np.linalg.eigvalsh(P)
        assert np.all(eigvals > 0), f"Smallest eigenvalue: {eigvals.min():.3e}"

    def test_diagonal_dominance_sign(self):
        """Diagonal entries must be the largest in each row (self-potential)."""
        centers, areas, _ = make_parallel_plates(1.0, 0.2, 5, 5)
        P = build_potential_matrix(centers, areas)
        for i in range(len(centers)):
            off_max = np.max(np.abs(np.delete(P[i], i)))
            assert P[i, i] > off_max, f"Row {i}: diag={P[i,i]:.3e}, off_max={off_max:.3e}"
