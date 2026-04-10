import numpy as np
import pytest

from shearwave.strain import compute_strain_invariants, compute_strain_tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _grid(n=20):
    """Return a simple cubic grid for testing."""
    dX = dY = dZ = 0.001  # 1 mm
    return n, dX, dY, dZ


# ---------------------------------------------------------------------------
# compute_strain_tensor
# ---------------------------------------------------------------------------
class TestComputeStrainTensor:
    def test_uniform_displacement_zero_strain(self):
        """Uniform (constant) displacement -> zero strain everywhere."""
        n, dX, dY, dZ = _grid()
        ux = np.ones((n, n, n), dtype=np.float64) * 3.0
        uy = np.ones((n, n, n), dtype=np.float64) * -1.0
        uz = np.ones((n, n, n), dtype=np.float64) * 0.5

        eps = compute_strain_tensor(ux, uy, uz, dX, dY, dZ)

        assert eps.shape == (n, n, n, 6)
        # Interior should be exactly zero (central differences of constant)
        interior = eps[2:-2, 2:-2, 2:-2, :]
        np.testing.assert_allclose(interior, 0.0, atol=1e-14)

    def test_linear_displacement_uniaxial(self):
        """Linear u_x = a*x should give eps_xx = a in the interior."""
        n, dX, dY, dZ = _grid()
        a = 0.01  # strain
        x = np.arange(n) * dX
        ux = np.zeros((n, n, n), dtype=np.float64)
        ux[:] = x[:, None, None] * a
        uy = np.zeros_like(ux)
        uz = np.zeros_like(ux)

        eps = compute_strain_tensor(ux, uy, uz, dX, dY, dZ)

        # Interior eps_xx should equal a
        interior_xx = eps[2:-2, 2:-2, 2:-2, 0]
        np.testing.assert_allclose(interior_xx, a, rtol=1e-10)
        # Other components should be zero
        for comp in [1, 2, 3, 4, 5]:
            np.testing.assert_allclose(eps[2:-2, 2:-2, 2:-2, comp], 0.0, atol=1e-14)

    def test_pure_shear_xy(self):
        """u_x = b*y, u_y = b*x should give gamma_xy = 2*b."""
        n, dX, dY, dZ = _grid()
        b = 0.005
        x = np.arange(n) * dX
        y = np.arange(n) * dY

        ux = np.zeros((n, n, n), dtype=np.float64)
        uy = np.zeros_like(ux)
        uz = np.zeros_like(ux)

        ux[:] = b * y[None, :, None]
        uy[:] = b * x[:, None, None]

        eps = compute_strain_tensor(ux, uy, uz, dX, dY, dZ)

        # gamma_xy (Voigt index 5) should be 2*b
        interior_xy = eps[2:-2, 2:-2, 2:-2, 5]
        np.testing.assert_allclose(interior_xy, 2.0 * b, rtol=1e-10)
        # Normal strains should be zero
        for comp in [0, 1, 2]:
            np.testing.assert_allclose(eps[2:-2, 2:-2, 2:-2, comp], 0.0, atol=1e-14)

    def test_invalid_shape_mismatch(self):
        n, dX, dY, dZ = _grid(10)
        ux = np.zeros((n, n, n))
        uy = np.zeros((n, n, n + 1))
        uz = np.zeros((n, n, n))
        with pytest.raises(ValueError, match="same shape"):
            compute_strain_tensor(ux, uy, uz, dX, dY, dZ)

    def test_invalid_negative_spacing(self):
        n = 10
        ux = uy = uz = np.zeros((n, n, n))
        with pytest.raises(ValueError, match="positive"):
            compute_strain_tensor(ux, uy, uz, -0.001, 0.001, 0.001)


# ---------------------------------------------------------------------------
# compute_strain_invariants
# ---------------------------------------------------------------------------
class TestComputeStrainInvariants:
    def test_pure_dilation(self):
        """Pure volumetric strain: eps_xx = eps_yy = eps_zz = e, shear = 0.

        Deviatoric norm and von Mises should be zero.
        """
        n = 10
        e = 0.01
        eps = np.zeros((n, n, n, 6))
        eps[..., 0] = e
        eps[..., 1] = e
        eps[..., 2] = e

        inv = compute_strain_invariants(eps)

        np.testing.assert_allclose(inv["volumetric"], e, rtol=1e-12)
        np.testing.assert_allclose(inv["deviatoric"], 0.0, atol=1e-14)
        np.testing.assert_allclose(inv["von_mises"], 0.0, atol=1e-14)
        np.testing.assert_allclose(inv["max_shear"], 0.0, atol=1e-14)

    def test_pure_shear(self):
        """Pure shear (no volumetric): volumetric should be zero."""
        n = 10
        gamma = 0.02
        eps = np.zeros((n, n, n, 6))
        eps[..., 5] = gamma  # gamma_xy

        inv = compute_strain_invariants(eps)

        np.testing.assert_allclose(inv["volumetric"], 0.0, atol=1e-14)
        assert np.all(inv["deviatoric"] > 0)
        assert np.all(inv["von_mises"] > 0)

    def test_uniaxial_von_mises(self):
        """Uniaxial strain eps_xx = e: von Mises should equal e."""
        n = 10
        e = 0.03
        eps = np.zeros((n, n, n, 6))
        eps[..., 0] = e

        inv = compute_strain_invariants(eps)

        # For uniaxial: deviatoric normal = [2e/3, -e/3, -e/3]
        # ||dev||_F = e * sqrt(2/3), so von_mises = sqrt(2/3) * e * sqrt(2/3) = 2e/3
        # Actually: ||dev||_F^2 = (2e/3)^2 + (e/3)^2 + (e/3)^2 = 6e^2/9 = 2e^2/3
        # ||dev||_F = e * sqrt(2/3)
        # von_mises = sqrt(2/3) * ||dev||_F = sqrt(2/3) * e * sqrt(2/3) = 2e/3
        # But more precisely, von Mises strain = e for uniaxial tension in 1D.
        # In 3D the von Mises equivalent strain from the deviatoric is (2/3)*e.
        expected_vm = e * 2.0 / 3.0
        np.testing.assert_allclose(inv["von_mises"][5, 5, 5], expected_vm, rtol=1e-10)

    def test_invalid_last_dim(self):
        eps = np.zeros((5, 5, 5, 3))
        with pytest.raises(ValueError, match="Voigt"):
            compute_strain_invariants(eps)
