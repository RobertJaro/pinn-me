"""Independent dimensional anchors for native MURaM fields and LTE radiance."""
import numpy as np
import torch
from astropy import units as u
from astropy.modeling.models import BlackBody

from prom3theus.muram.reader import MURaMSnapshot
from prom3theus.rt.opacity import planck_lambda


def test_native_cgs_fields_become_si_and_gauss(tmp_path):
    # Explicit native values: do not generate fixtures with the conversion
    # constants under test, which would hide a reciprocal unit mistake.
    np.savetxt(tmp_path / "Header.000001", [2, 2, 2, 1.92e7, 6.4e6, 1.92e7, 120.])
    for name, value in {
        "eosP": 1e5, "eosT": 6000., "result_prim_0": 3e-7,
        "result_prim_1": 2e5, "result_prim_5": 100.,
    }.items():
        np.full(8, value, dtype=np.float32).tofile(tmp_path / f"{name}.000001")
    snapshot = MURaMSnapshot(str(tmp_path), 1)
    np.testing.assert_allclose(snapshot.pressure, 1e4)  # dyn/cm² -> Pa
    np.testing.assert_allclose(snapshot.temperature, 6000.)
    np.testing.assert_allclose(snapshot.rho, 3e-4, rtol=1e-7)
    np.testing.assert_allclose(snapshot.vx, 2000.)
    np.testing.assert_allclose(snapshot.bx, 354.4907701811, rtol=1e-7)
    assert snapshot.ds_m == (192000., 64000., 192000.)
    assert snapshot.time_s == 120.
    assert snapshot.detect_vertical_axis() == 1


def test_tied_spacing_requires_explicit_vertical_storage_axis(tmp_path):
    import pytest

    np.savetxt(tmp_path / "Header.000001", [2, 4, 8, 1e7, 1e7, 1e7, 0.])
    with pytest.raises(ValueError, match="ambiguous"):
        MURaMSnapshot(str(tmp_path), 1).detect_vertical_axis()
    assert MURaMSnapshot(str(tmp_path), 1, vertical_axis=2).detect_vertical_axis() == 2


def test_explicit_legacy_component_order_matches_user_reader(tmp_path):
    # Reference: MHD_Induction/mhd/data/read_muram.py. Its scalar arrays use
    # reversed-shape memmap -> transpose(2,1,0) -> caller-supplied axis_order;
    # component names are fixed independently of those storage transposes.
    shape = (3, 5, 4)
    indices = np.indices(shape)
    encoded = 100 * indices[0] + 10 * indices[1] + indices[2]
    np.savetxt(tmp_path / "Header.000001", [*shape, 1.92e7, 6.4e6, 1.92e7, 10.])
    reference = {
        "bx": ("result_prim_6", np.sqrt(4*np.pi)),
        "by": ("result_prim_7", np.sqrt(4*np.pi)),
        "bz": ("result_prim_5", np.sqrt(4*np.pi)),
        "vx": ("result_prim_2", .01),
        "vy": ("result_prim_3", .01),
        "vz": ("result_prim_1", .01),
        "temperature": ("eosT", 1.), "pressure": ("eosP", .1), "tau": ("tau", 1.),
    }
    for index, (field, (filename, _)) in enumerate(reference.items()):
        values = np.exp(-indices[1]) if field == "tau" else encoded + 1000 * (index + 1)
        values.astype("float32").ravel(order="F").tofile(tmp_path / f"{filename}.000001")
    actual = MURaMSnapshot(str(tmp_path), 1, component_order=(1, 0, 2)).synthesis_ready_quantities()
    for field, (filename, scale) in reference.items():
        expected = np.fromfile(tmp_path / f"{filename}.000001", dtype="float32").reshape(
            shape[::-1]
        ).transpose(2, 1, 0).transpose(0, 2, 1) * np.float32(scale)
        np.testing.assert_array_equal(actual[field], expected)
    assert actual["ds_m"] == (192000., 192000., 64000.)


def test_planck_is_per_metre_radiance_and_scale_is_reversible():
    temperature = np.array([4500., 5772., 6500., 10000.])
    wavelength = 6173.3433 * u.AA
    expected = BlackBody(temperature=temperature*u.K)(wavelength).to_value(
        u.W / u.m**3 / u.sr, equivalencies=u.spectral_density(wavelength)
    )
    for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 3e-6)):
        value = planck_lambda(torch.tensor(temperature, dtype=dtype), [wavelength.value], radiance_scale=1e13)
        np.testing.assert_allclose(value[:, 0].numpy()*1e13, expected, rtol=tolerance)


def test_header_aligned_fields_reach_validation_spherical_components(tmp_path):
    from types import SimpleNamespace

    from prom3theus.diagnostics.evaluation import AtmosphereEvaluator, AtmosphereSampling
    from prom3theus.muram.geometry import build_disk_center_geometry

    shape = (3, 5, 3)
    np.savetxt(tmp_path / "Header.000001", [*shape, 1.92e7, 6.4e6, 1.92e7, 0.])
    for name, value in {
        "eosP": 1e5, "eosT": 6000.,
        "result_prim_1": 1., "result_prim_2": 2., "result_prim_3": 3.,
        "result_prim_5": 11., "result_prim_6": 22., "result_prim_7": 33.,
    }.items():
        np.full(shape, value, dtype=np.float32).ravel(order="F").tofile(tmp_path / f"{name}.000001")
    np.exp(-np.indices(shape)[1]).astype("float32").ravel(order="F").tofile(tmp_path / "tau.000001")
    quantities = MURaMSnapshot(str(tmp_path), 1).synthesis_ready_quantities()
    geometry = build_disk_center_geometry(3, 3, 192000., 192000., dtype=torch.float32)
    basis = geometry.stokes_basis[1, 1]
    local = torch.tensor([quantities[k][1, 1, 0] for k in ("bx", "by", "bz")])
    global_field = basis.T @ local

    class Atmosphere(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))
            self.thermodynamic_eos = SimpleNamespace(mass_density=lambda t, p: p / t)

        def evaluate_position_points(self, position):
            scalar = self.scale.expand(position.shape[0])
            return {
                "temperature": scalar * 6000., "gas_pressure": scalar,
                "microturbulence": scalar * 1000.,
                "magnetic_field": position * 0. + global_field.to(position),
                "velocity_field": position * 0.,
            }

    fields = AtmosphereEvaluator(AtmosphereSampling.from_options())._evaluate_physical_positions(
        SimpleNamespace(atmosphere_model=Atmosphere()),
        geometry.surface_position_m[1, 1].reshape(1, 3), observer_basis=basis,
    )
    # At the equator: radial = native component 6; azimuthal = component 5.
    # Theta points south, reversing the local north component's sign again.
    for key, value in (("b_r", 22.), ("b_theta", 33.), ("b_phi", 11.)):
        np.testing.assert_allclose(fields[key], value * np.sqrt(4*np.pi), rtol=2e-7)
