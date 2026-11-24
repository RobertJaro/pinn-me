"""Validate synthesis requests before creating or replacing output products."""

import inspect
import math
from numbers import Integral

import torch

from prom3theus.instruments.aia_euv.operator import AIAEmissionOperator
from prom3theus.instruments.registry import build_instrument
from prom3theus.validation.reference_synthesis import LTESynthesizer

from .hmi_filter import build_idealized_hmi_filter
from .reader import MURaMSnapshot, _REQUIRED_FOR_SYNTHESIS
from .synthesize import AIA_RESPONSE_RESOURCE, DEFAULT_AIA_CHANNELS
from .atmosphere_builder import DEFAULT_STOKES_HEIGHT_MEGAMETERS, stokes_depth_count


def validate_options(synthesize, options, *, kind):
    """Check instrument construction and spectral sampling without reading cubes."""
    allowed = ({"channels_angstrom", "response_resource"} if kind == "aia" else
               {"instrument", "observed_wavelength_angstrom", "line_ids", "response"})
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unsupported synthesis options: {sorted(unknown)}")
    inspect.signature(synthesize).bind(None, **options)
    if kind == "aia":
        AIAEmissionOperator(options.get("channels_angstrom", DEFAULT_AIA_CHANNELS),
                            response_resource=options.get("response_resource", AIA_RESPONSE_RESOURCE))
        return
    if "instrument" not in options:
        build_idealized_hmi_filter()
        LTESynthesizer(line_ids=("FeI_6173.3352",))
        return
    operator = build_instrument(options["instrument"]).double()
    observed = torch.as_tensor(options["observed_wavelength_angstrom"], dtype=torch.float64)
    if (observed.ndim != 1 or observed.numel() < 2 or not torch.isfinite(observed).all()
            or torch.any(observed <= 0) or not torch.all(observed[1:] > observed[:-1])):
        raise ValueError("Observed wavelengths must be finite, positive and strictly increasing.")
    wavelength = operator.synthesis_grid(observed)
    if callable(getattr(operator, "prepare", None)):
        operator.prepare(wavelength, observed)
    response = {name: torch.as_tensor(value, dtype=torch.float64)
                for name, value in (options.get("response") or {}).items()}
    operator(torch.zeros((1, 4, wavelength.numel()), dtype=torch.float64), wavelength, observed,
             **{name: value.expand(1, *value.shape) for name, value in response.items()})
    LTESynthesizer(line_ids=tuple(options["line_ids"]))


def validate_execution(args, *, uniform_shell_grid=None):
    """Check execution settings and cheap source metadata before destructive replacement."""

    height_megameters = getattr(args, "stokes_height_megameters", DEFAULT_STOKES_HEIGHT_MEGAMETERS)
    if height_megameters is not None and (isinstance(height_megameters, bool) or not math.isfinite(height_megameters) or height_megameters <= 0):
        raise ValueError("stokes_height_megameters must be finite and positive, or None for full height.")
    for name in ("batch_size", "stride", "binning"):
        value = getattr(args, name, None)
        if value is None and name != "batch_size":
            continue
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not args.iterations or len(set(args.iterations)) != len(args.iterations):
        raise ValueError("iterations must be nonempty and unique.")
    if not args.source_path.is_dir():
        raise ValueError(f"Source directory not found: {args.source_path}")
    # Normal runs remain sequential and can preserve earlier snapshots if a
    # later one is missing. Before --overwrite, check every requested header and
    # field filename so a malformed replacement cannot erase a good output.
    if args.overwrite and args.out.exists():
        for iteration in args.iterations:
            snapshot = MURaMSnapshot(str(args.source_path), iteration,
                                     vertical_axis=args.vertical_axis,
                                     component_order=tuple(args.component_order))
            snapshot.require_quantities(_REQUIRED_FOR_SYNTHESIS)
            vertical = snapshot.detect_vertical_axis()
            if uniform_shell_grid is None:
                stokes_depth_count(snapshot.shape[vertical], snapshot.ds_m[vertical], height_megameters)
            else:
                uniform_shell_grid.retained_depth(snapshot.shape[vertical], snapshot.ds_m[vertical])
            step, factor = args.stride or 1, args.binning or 1
            horizontal = [size for axis, size in enumerate(snapshot.shape) if axis != vertical]
            if any(((size + step - 1) // step) % factor for size in horizontal):
                raise ValueError("Horizontal dimensions after stride must be divisible by binning.")
