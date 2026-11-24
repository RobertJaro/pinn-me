"""Eager volume -> true rays -> polarized spectra -> saved evaluation arrays."""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import yaml

from prom3theus.instruments.registry import build_instrument
from prom3theus.muram.ray_volume import EagerVolume
from prom3theus.muram.synthesize import synthesize_hmi, synthesize_stokes
from prom3theus.muram.synthesize_rays import load_ray_product, synthesize_ray_volume


def _snapshot(shape=(3, 3, 15), dx=100.):
    fields = {name: np.zeros(shape, dtype=np.float32)
              for name in ("bx", "by", "bz", "vx", "vy", "vz")}
    fields.update(temperature=np.broadcast_to(np.linspace(6500, 4500, shape[-1], dtype=np.float32), shape).copy(),
                  pressure=np.broadcast_to(np.geomspace(2e6, 10, shape[-1], dtype=np.float32), shape).copy(),
                  tau=np.broadcast_to(np.geomspace(100, 1e-6, shape[-1], dtype=np.float32), shape).copy(),
                  ds_m=(dx, dx, 1e4))
    fields["bx"][:] = 300.
    fields["by"][:] = 450.
    fields["bz"][:] = 900.
    return SimpleNamespace(time_s=0., synthesis_ready_quantities=lambda: fields)


def test_disk_center_ray_reproduces_column_and_saved_spectral_response(tmp_path):
    snapshot = _snapshot()
    reference = synthesize_hmi(snapshot, dtype=torch.float64, progress=False,
                               keep_raw_stokes=True, diagnostics=False, device="cpu")
    volume = EagerVolume.from_snapshot(snapshot, progress=False)
    output = synthesize_ray_volume(volume, tmp_path / "product", bottom_height_megameters=0.,
                                    shell_height_bounds_megameters=(0., .14), depth_samples=15,
                                    workers=0, progress=False, dtype=torch.float64, device="cpu")
    arrays, metadata = load_ray_product(output)
    assert arrays["stokes"].shape == (3, 3, 4, 163)
    assert arrays["instrument_stokes"].shape == (3, 3, 4, 6)
    assert arrays["valid_mask"].all()
    assert (arrays["boundary_type"] == 1).all()
    # At the central pixel, both spherical intersections coincide exactly with
    # native cell centers. Off-axis intersections correctly differ slightly.
    np.testing.assert_allclose(arrays["stokes"][1, 1] / 1e13,
                               reference.raw_stokes[1, 1].numpy(), rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(arrays["instrument_stokes"][1, 1] / 1e13,
                               reference.stokes[1, 1].numpy(), rtol=2e-7, atol=2e-8)
    options = metadata["instrument_options"]
    operator = build_instrument(options["instrument"]).double()
    observed = torch.from_numpy(np.array(arrays["observed_wavelength_angstrom"]))
    wavelength = torch.from_numpy(np.array(arrays["wavelength_angstrom"]))
    response = {name: torch.as_tensor(value, dtype=torch.float64).expand(3, 3, *np.shape(value))
                for name, value in options["response"].items()}
    filtered = operator(torch.from_numpy(np.array(arrays["stokes"])), wavelength, observed, **response)
    np.testing.assert_allclose(arrays["instrument_stokes"], filtered.numpy(), rtol=1e-12, atol=1e-3)
    assert metadata["units"]["stokes"] == "W m^-3 sr^-1"
    assert arrays["stokes"].flags.writeable is False
    np.testing.assert_array_equal(arrays["detector_basis"], [[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    np.testing.assert_allclose(arrays["synthesis_stokes_basis"], [[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]], atol=1e-15)
    assert metadata["polarization_convention"]["magnetic_azimuth_offset_deg"] == 90.


def test_inclined_rays_sample_horizontal_structure_and_flag_vacuum_boundaries(tmp_path):
    snapshot = _snapshot((5, 3, 15), dx=1e5)
    fields = snapshot.synthesis_ready_quantities()
    fields["temperature"] += np.arange(5, dtype=np.float32)[:, None, None] * 500.
    volume = EagerVolume.from_snapshot(snapshot, progress=False)
    common = dict(bottom_height_megameters=0., depth_samples=64, workers=0,
                  progress=False, dtype=torch.float64, device="cpu")
    straight = synthesize_ray_volume(volume, tmp_path / "straight", **common)
    inclined = synthesize_ray_volume(volume, tmp_path / "inclined", observer_direction=(1., .5, .3), **common)
    vertical_arrays, _ = load_ray_product(straight)
    tilted_arrays, _ = load_ray_product(inclined)
    center = tuple(size // 2 for size in tilted_arrays["valid_mask"].shape)
    assert tilted_arrays["valid_mask"][center]
    assert tilted_arrays["boundary_type"][center] == 1
    center_stokes = tilted_arrays["instrument_stokes"][center]
    vertical_stokes = vertical_arrays["instrument_stokes"][1, 2]
    assert np.max(np.abs(center_stokes - vertical_stokes)) > .001 * np.max(np.abs(vertical_stokes))
    missed = ~tilted_arrays["valid_mask"]
    assert missed.any()
    assert (tilted_arrays["stokes"][missed] == 0).all()
    assert (tilted_arrays["instrument_stokes"][missed] == 0).all()
    assert (tilted_arrays["path_length_m"][missed] == 0).all()
    assert np.any(tilted_arrays["boundary_type"] == 2)


def test_packaged_hinode_ray_spectra_match_column_reference(tmp_path):
    options = yaml.safe_load(Path("configs/muram/hinode/synthesis.yaml").read_text())
    options["observed_wavelength_angstrom"] = np.asarray(options["observed_wavelength_angstrom"])
    snapshot = _snapshot()
    reference = synthesize_stokes(snapshot, **options, dtype=torch.float64, progress=False,
                                  diagnostics=False, device="cpu")
    volume = EagerVolume.from_snapshot(snapshot, progress=False)
    output = synthesize_ray_volume(volume, tmp_path / "hinode", options=options, bottom_height_megameters=0.,
                                    shell_height_bounds_megameters=(0., .14), depth_samples=15,
                                    workers=0, progress=False, dtype=torch.float64, device="cpu")
    arrays, metadata = load_ray_product(output)
    assert arrays["instrument_stokes"].shape == (3, 3, 4, 112)
    np.testing.assert_allclose(arrays["instrument_stokes"][1, 1] / 1e13,
                               reference.stokes[1, 1].numpy(), rtol=2e-7, atol=2e-8)
    np.testing.assert_array_equal(arrays["synthesis_stokes_basis"], arrays["detector_basis"])
    assert metadata["polarization_convention"]["magnetic_azimuth_offset_deg"] == 0.
