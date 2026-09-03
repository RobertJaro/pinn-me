import pytest
import torch

from prom3theus.core import SPEED_OF_LIGHT
from prom3theus.rt import AtomicDatabase, PolarizedLineOpacity


def _line_problem(*, samples: int = 401):
    dtype = torch.float64
    atomic = AtomicDatabase()
    line = atomic.get_line("FeI_6302.4932")
    opacity = PolarizedLineOpacity((line,), atomic)
    wavelength = torch.linspace(6302.25, 6302.75, samples, dtype=dtype)
    temperature = torch.tensor([6000.0], dtype=dtype)
    microturbulence = torch.tensor([1000.0], dtype=dtype)
    continuum = torch.full((1, samples), 1.0e-7, dtype=dtype)
    alpha500 = torch.tensor([1.0e-7], dtype=dtype)
    populations = {line.id: torch.tensor([1.0e14], dtype=dtype)}
    hydrogen = torch.tensor([1.0e22], dtype=dtype)
    return {
        "opacity": opacity,
        "wavelength": wavelength,
        "temperature": temperature,
        "microturbulence": microturbulence,
        "continuum_extinction": continuum,
        "alpha500": alpha500,
        "lower_level_populations": populations,
        "damping_electron_density": None,
        "damping_hydrogen_neutral": hydrogen,
        "normalize_to_alpha500": True,
    }


def _propagation(problem, *, velocity_los=0.0, magnetic_field=(0.0, 0.0, 0.0)):
    return problem["opacity"](
        problem["wavelength"],
        problem["temperature"],
        problem["temperature"].new_tensor([velocity_los]),
        problem["microturbulence"],
        problem["temperature"].new_tensor([magnetic_field]),
        problem["continuum_extinction"],
        problem["alpha500"],
        lower_level_populations=problem["lower_level_populations"],
        damping_electron_density=problem["damping_electron_density"],
        damping_hydrogen_neutral=problem["damping_hydrogen_neutral"],
        normalize_to_alpha500=problem["normalize_to_alpha500"],
    )


def test_positive_los_velocity_redshifts_the_line_opacity_peak():
    problem = _line_problem(samples=1001)
    blue = _propagation(problem, velocity_los=-3000.0)[0, :, 0, 0]
    red = _propagation(problem, velocity_los=3000.0)[0, :, 0, 0]
    wavelength = problem["wavelength"]

    assert wavelength[red.argmax()] > wavelength[blue.argmax()]


def test_magnetic_reversal_and_quarter_turn_obey_stokes_conventions():
    problem = _line_problem()
    positive_los = _propagation(problem, magnetic_field=(0.0, 0.0, 1200.0))
    negative_los = _propagation(problem, magnetic_field=(0.0, 0.0, -1200.0))
    torch.testing.assert_close(positive_los[..., 0, 0], negative_los[..., 0, 0])
    torch.testing.assert_close(positive_los[..., 0, 3], -negative_los[..., 0, 3])

    positive_q = _propagation(problem, magnetic_field=(1200.0, 0.0, 0.0))
    positive_u = _propagation(problem, magnetic_field=(0.0, 1200.0, 0.0))
    torch.testing.assert_close(positive_q[..., 0, 0], positive_u[..., 0, 0])
    torch.testing.assert_close(positive_q[..., 0, 1], -positive_u[..., 0, 1])


def test_polarized_opacity_rejects_nonphysical_inputs():
    problem = _line_problem(samples=3)
    with pytest.raises(ValueError, match="strictly subluminal"):
        _propagation(problem, velocity_los=SPEED_OF_LIGHT)

    problem["damping_hydrogen_neutral"] = torch.tensor(
        [float("nan")], dtype=torch.float64
    )
    with pytest.raises(ValueError, match="damping_hydrogen_neutral"):
        _propagation(problem)

    problem = _line_problem(samples=3)
    problem["continuum_extinction"] = torch.ones(1, 3, dtype=torch.float32)
    with pytest.raises(TypeError, match="dtype and device"):
        _propagation(problem)
