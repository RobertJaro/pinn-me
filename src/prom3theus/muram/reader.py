"""Memmapped access to raw MURaM 3D output cubes.

MURaM writes each physical quantity as a flat binary file named
``<field_id>.<iteration>`` alongside a small ``Header.<iteration>`` text file
holding the grid shape, cell spacing, and simulation time. This module reads
those files and converts every quantity to SI units (with magnetic field kept
in gauss, matching :class:`prom3theus.rt.atmosphere.StratifiedAtmosphere`).

The on-disk axis order and unit conventions mirror the MHD_Induction project's
``mhd.data.read_muram.MURaMSnapshot`` reader (not a runtime dependency here;
duplicated to keep this package self-contained).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path

import numpy as np

from .columns import NativeColumnDataset, NativeColumns

# Raw-file id for each stored native MURaM component.
# synthesis_ready_quantities maps these to the local horizontal-x/y, vertical-z frame. Values marked
# "code units" require the sqrt(4 pi) Lorentz-Heaviside-to-gauss conversion
# applied by MURaM convention; everything else is read directly in CGS and
# converted to SI below.
_FIELD_IDS = {
    "bz": "result_prim_7",
    "by": "result_prim_6",
    "bx": "result_prim_5",
    "vz": "result_prim_3",
    "vy": "result_prim_2",
    "vx": "result_prim_1",
    "rho": "result_prim_0",
    "pressure": "eosP",
    "temperature": "eosT",
    "tau": "tau",
    "qtot": "Qtot",
}

_SQRT_4PI = float(np.sqrt(4.0 * np.pi))

# CGS -> SI conversion factors applied after loading each raw field. Magnetic
# field is left in gauss (its CGS unit) since that is what the LTE synthesis
# pipeline expects.
_UNIT_SCALE = {
    "bz": _SQRT_4PI,
    "by": _SQRT_4PI,
    "bx": _SQRT_4PI,
    "vz": 1.0e-2,  # cm/s -> m/s
    "vy": 1.0e-2,
    "vx": 1.0e-2,
    "rho": 1.0e3,  # g/cm^3 -> kg/m^3
    "pressure": 1.0e-1,  # erg/cm^3 -> Pa
    "temperature": 1.0,  # K
    "tau": 1.0,  # dimensionless
    "qtot": 1.0e-1,  # erg/(cm^3 s) -> W/m^3
}

_REQUIRED_FOR_SYNTHESIS = (
    "bx",
    "by",
    "bz",
    "vx",
    "vy",
    "vz",
    "pressure",
    "temperature",
    "tau",
)


NATIVE_REFERENCE_SCHEMA = "muram_native_reference_v1"
_SOURCE_FIELDS = {"header": "Header", **{
    name: _FIELD_IDS[name] for name in _REQUIRED_FOR_SYNTHESIS
}}


@dataclass(frozen=True)
class MURaMSnapshot:
    """One MURaM iteration, read into SI-unit numpy arrays.

    Individual field attributes return arrays in the header's native
    ``(dim0, dim1, dim2)`` order, permuted by ``axis_order``. MURaM does not use a fixed
    axis convention across runs (which native axis is vertical differs
    between simulations, and even between snapshots of the same run family
    seen so far), so this class does not guess it. Use
    :meth:`detect_vertical_axis` to identify the finest-spaced storage axis, and :meth:`synthesis_ready_quantities` (which calls
    it automatically) to get arrays reordered to ``[dim_a, dim_b, z]``.

    Synthesis uses :meth:`synthesis_columns`: a lazy mapped dataset and a
    two-worker DataLoader. Explicit field access and
    :meth:`synthesis_ready_quantities` retain the eager array API.

    Parameters
    ----------
    source_path:
        Directory containing ``Header.<iteration>`` and the field files.
    iteration:
        MURaM iteration number (matches the file suffix, zero-padded or not
        depending on the run's own naming).
    axis_order:
        Permutation applied to the on-disk ``(dim0, dim1, dim2)`` axes before
        any vertical-axis detection. Defaults to the identity (no reorder).
    vertical_axis:
        Explicit vertical storage axis after axis_order, or None for the finest spacing.
    component_order:
        Stored vector component indices aligned with original header axes 0/1/2.
        Default (0,1,2) aligns primitive 1/2/3 and 5/6/7 with those axes.
    """

    source_path: str
    iteration: int
    axis_order: tuple[int, int, int] = (0, 1, 2)
    vertical_axis: int | None = None
    component_order: tuple[int, int, int] = (0, 1, 2)
    shape: tuple[int, int, int] = field(init=False)
    ds_m: tuple[float, float, float] = field(init=False)
    time_s: float = field(init=False)
    _native_shape: tuple[int, int, int] = field(init=False, repr=False, compare=False)
    _arrays: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _mappings: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _synthesis_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if sorted(self.component_order) != [0, 1, 2]:
            raise ValueError("component_order must be a permutation of (0, 1, 2).")
        if sorted(self.axis_order) != [0, 1, 2]:
            raise ValueError("axis_order must be a permutation of (0, 1, 2).")
        header_path = self._field_path("Header")
        if not os.path.exists(header_path):
            raise FileNotFoundError(f"MURaM header file not found: {header_path}")
        if self.vertical_axis is not None and self.vertical_axis not in (0, 1, 2):
            raise ValueError("vertical_axis must be 0, 1, 2, or None (auto).")
        header = np.loadtxt(header_path, dtype=np.float64).reshape(-1)
        if (
            header.size < 7
            or not np.isfinite(header[:7]).all()
            or np.any(header[:3] < 2)
            or np.any(header[:3] != np.floor(header[:3]))
            or np.any(header[3:6] <= 0)
        ):
            raise ValueError(
                "MURaM Header requires three grid sizes >=2, positive spacing, and finite time."
            )
        native_shape = tuple(int(v) for v in header[0:3])
        native_ds_cm = tuple(float(v) for v in header[3:6])
        object.__setattr__(self, "_native_shape", native_shape)
        object.__setattr__(
            self,
            "shape",
            tuple(native_shape[axis] for axis in self.axis_order),
        )
        object.__setattr__(
            self,
            "ds_m",
            tuple(native_ds_cm[axis] * 1.0e-2 for axis in self.axis_order),
        )
        object.__setattr__(self, "time_s", float(header[6]))

    def _field_path(self, field_id: str) -> str:
        for suffix in (str(self.iteration), f"{self.iteration:06d}"):
            path = os.path.join(self.source_path, f"{field_id}.{suffix}")
            if os.path.exists(path):
                return path
        return os.path.join(self.source_path, f"{field_id}.{self.iteration}")

    def available_quantities(self) -> set[str]:
        return {
            name
            for name, field_id in _FIELD_IDS.items()
            if os.path.exists(self._field_path(field_id))
        }

    def require_quantities(self, quantities) -> None:
        missing = sorted(set(quantities) - self.available_quantities())
        if missing:
            missing_files = [f"{_FIELD_IDS[name]}.{self.iteration}" for name in missing]
            raise FileNotFoundError(
                f"Missing MURaM output files in {self.source_path!r} for iteration "
                f"{self.iteration}: {missing_files}"
            )

    def __getattr__(self, name: str):
        if name not in _FIELD_IDS:
            raise AttributeError(name)
        return self._load(name)

    def _load(self, name: str) -> np.ndarray:
        cached = self._arrays.get(name)
        if cached is not None:
            return cached
        array = np.array(self._mapped(name), dtype=np.float32, copy=True, order="K")
        np.multiply(array, np.float32(_UNIT_SCALE[name]), out=array)
        self._arrays[name] = array
        return array

    def _mapped(self, name: str) -> np.ndarray:
        cached = self._mappings.get(name)
        if cached is not None:
            return cached
        field_id = _FIELD_IDS[name]
        path = self._field_path(field_id)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"MURaM output file not found: {path}. Available quantities: "
                f"{sorted(self.available_quantities())}"
            )
        # MURaM stores each cube with its first header axis varying fastest,
        # i.e. C-contiguous in reverse of the header's own (dim0, dim1, dim2)
        # order. Reading with the reversed shape and transposing back to
        # (dim0, dim1, dim2) recovers the header's native axis order before
        # applying axis_order to reach the requested (x, y, z) layout.
        raw = np.memmap(
            path, mode="r", dtype=np.float32, shape=tuple(reversed(self._native_shape))
        )
        native = np.transpose(raw, (2, 1, 0))
        array = np.transpose(native, self.axis_order)
        self._mappings[name] = array
        return array

    def detect_vertical_axis(self) -> int:
        """Locate physical z by the finest native cell spacing.

        Isotropic or tied finest spacings require an explicit storage axis.
        Component alignment is controlled independently by component_order;
        do not assume primitive 1/5 is vertical for every exported cube.
        """
        if self.vertical_axis is not None:
            return self.vertical_axis
        spacing = np.asarray(self.ds_m)
        finest = np.flatnonzero(np.isclose(spacing, spacing.min(), rtol=1e-6, atol=0))
        if finest.size != 1:
            raise ValueError("Finest MURaM spacing is ambiguous; specify --vertical-axis.")
        return int(finest[0])

    def _synthesis_layout(self):
        if "layout" in self._synthesis_cache:
            return self._synthesis_cache["layout"]
        self.require_quantities(_REQUIRED_FOR_SYNTHESIS)
        vertical_axis = self.detect_vertical_axis()
        # tau decreases with height; MURaM's vertical index direction
        # (increasing or decreasing height) is likewise not guaranteed, so
        # orient explicitly using the tau profile rather than assuming index
        # order matches height order.
        other_axes = tuple(a for a in range(3) if a != vertical_axis)
        # Only the two boundary means determine orientation. Read those planes
        # rather than scanning the entire tau cube before batch loading starts.
        tau = self._mapped("tau")
        boundary = [slice(None)] * 3
        boundary[vertical_axis] = 0
        first = tau[tuple(boundary)].mean(dtype=np.float64)
        boundary[vertical_axis] = -1
        last = tau[tuple(boundary)].mean(dtype=np.float64)
        if not np.isfinite([first, last]).all() or min(first, last) < 0 or first == last:
            raise ValueError(
                f"Cannot determine vertical orientation for MURaM iteration {self.iteration}: "
                "tau boundary means must be finite, non-negative, and distinct."
            )
        increasing_height = first > last

        # By default primitive components follow the original header axes.
        # Some exports permute arrays independently of vector components; those
        # require an explicit component_order, not a universal x-is-vertical rule.
        order = other_axes + (vertical_axis,)
        native_order = tuple(self.axis_order[a] for a in order)
        component_order = tuple(self.component_order[a] for a in native_order)
        parity = round(np.linalg.det(np.eye(3)[list(component_order)]))
        z_sign = 1 if increasing_height else -1
        y_sign = parity * z_sign
        ds_m = tuple(self.ds_m[axis] for axis in order)
        layout = (order, component_order, z_sign, y_sign, ds_m)
        self._synthesis_cache["layout"] = layout
        return layout

    def _oriented_fields(self):
        order, component_order, z_sign, y_sign, ds_m = self._synthesis_layout()
        quantities = {}
        for name in _REQUIRED_FOR_SYNTHESIS:
            source_name = name
            sign = 1
            if name[0] in ("b", "v") and name[1:] in ("x", "y", "z"):
                component = "xyz".index(name[1])
                source_name = name[0] + "xyz"[component_order[component]]
                sign = (1, y_sign, z_sign)[component]
            array = np.transpose(self._load(source_name), order)
            if z_sign < 0:
                array = np.flip(array, axis=-1)
            if y_sign < 0:
                array = np.flip(array, axis=1)
            quantities[name] = array if sign == 1 else -array
        return quantities, ds_m

    def synthesis_ready_quantities(self) -> dict[str, np.ndarray]:
        """Explicitly materialize and cache the full oriented SI atmosphere.

        Arrays have shape [horizontal x, horizontal y, upward z]. Synthesis
        itself uses synthesis_columns() to load only its required fields into a
        shared full-volume cache before sequential batch processing.
        """
        if "quantities" not in self._synthesis_cache:
            quantities, ds_m = self._oriented_fields()
            self._synthesis_cache["quantities"] = {**quantities, "ds_m": ds_m}
        return dict(self._synthesis_cache["quantities"])

    def synthesis_columns(self) -> NativeColumns:
        """Reuse native field descriptors and the eager synthesis-volume cache."""
        if "columns" not in self._synthesis_cache:
            order, component_order, z_sign, y_sign, ds_m = self._synthesis_layout()
            paths, scales, signs = {}, {}, {}
            for name in _REQUIRED_FOR_SYNTHESIS:
                source_name, sign = name, 1
                if name[0] in ("b", "v") and name[1:] in ("x", "y", "z"):
                    component = "xyz".index(name[1])
                    source_name = name[0] + "xyz"[component_order[component]]
                    sign = (1, y_sign, z_sign)[component]
                paths[name] = os.path.abspath(self._field_path(_FIELD_IDS[source_name]))
                scales[name], signs[name] = _UNIT_SCALE[source_name], sign
            dataset = NativeColumnDataset(
                paths, self._native_shape, tuple(self.axis_order[axis] for axis in order),
                scales, signs, y_sign=y_sign, z_sign=z_sign,
            )
            self._synthesis_cache["columns"] = NativeColumns(
                dataset, ds_m,
            )
        return self._synthesis_cache["columns"]

    def native_reference(self, *, stride=1, binning=1):
        """Describe the source and pixel mapping for later native-data evaluation."""
        return native_snapshot_reference(self, stride=stride, binning=binning)

    def close(self):
        """Release this snapshot's cached arrays and mappings before the next one."""
        columns = self._synthesis_cache.get("columns")
        if columns is not None:
            columns.close()
        self._synthesis_cache.clear()
        self._arrays.clear()
        self._mappings.clear()


def _sampling_factor(value, name):
    if value is None:
        return 1
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def native_snapshot_reference(snapshot, *, stride=1, binning=1):
    """Record source identity and resolved geometry without loading cube volumes."""
    stride = _sampling_factor(stride, "stride")
    binning = _sampling_factor(binning, "binning")
    order, components, z_sign, y_sign, spacing = snapshot._synthesis_layout()
    header_axes = tuple(snapshot.axis_order[axis] for axis in order)
    shape = tuple(snapshot._native_shape[axis] for axis in header_axes)
    selected = tuple((size + stride - 1) // stride for size in shape[:2])
    if any(size % binning for size in selected):
        raise ValueError("Horizontal dimensions after stride must be divisible by binning.")
    header_spacing = [0.0] * 3
    for axis, header_axis in enumerate(snapshot.axis_order):
        header_spacing[header_axis] = snapshot.ds_m[axis]
    files = {}
    for name, field_id in _SOURCE_FIELDS.items():
        path = Path(snapshot._field_path(field_id))
        stat = path.stat()
        files[name] = {
            "name": path.name,
            "path": str(path.resolve()),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }
    return {
        "schema": NATIVE_REFERENCE_SCHEMA,
        "source_path": str(Path(snapshot.source_path).resolve()),
        "iteration": int(snapshot.iteration),
        "time_s": float(snapshot.time_s),
        "identity_method": "file_metadata_v1; not a content checksum",
        "files": files,
        "reader": {
            "axis_order": [int(axis) for axis in snapshot.axis_order],
            "vertical_axis": None if snapshot.vertical_axis is None else int(snapshot.vertical_axis),
            "component_order": [int(axis) for axis in snapshot.component_order],
        },
        "native_geometry": {
            "header_shape": list(snapshot._native_shape),
            "header_spacing_m": header_spacing,
            "oriented_shape_xyz": list(shape),
            "oriented_spacing_m": list(spacing),
            "header_axes_for_local_xyz": [int(axis) for axis in header_axes],
            "vector_components_for_local_xyz": [int(axis) for axis in components],
            "index_direction_xyz": [1, y_sign, z_sign],
            "vector_sign_xyz": [1, y_sign, z_sign],
            "resolved_vertical_header_axis": int(header_axes[2]),
            "local_axes": ["solar west", "solar north", "upward"],
            "height_origin": "bottom native cell center; not the photosphere",
            "photosphere_height_m": None,
            "height_range_m": [0.0, (shape[2] - 1) * spacing[2]],
            "source_dtype": np.dtype(np.float32).str,
            "source_storage_order": "header axis 0 varies fastest",
        },
        "units": {
            "temperature": "K", "pressure": "Pa", "magnetic_field": "G",
            "velocity": "m s^-1; local co-rotating flow", "distance": "m", "tau": "1",
            "native_to_physical_scale": {
                name: _UNIT_SCALE[name] for name in _REQUIRED_FOR_SYNTHESIS
            },
        },
        "pixel_mapping": {
            "native_array_order": ["x", "y", "z"],
            "output_map_order": ["y", "x"],
            "stride": stride,
            "binning": binning,
            "output_shape_yx": [selected[1] // binning, selected[0] // binning],
            "native_x_indices": "(output_x * binning + subpixel_x) * stride",
            "native_y_indices": "(output_y * binning + subpixel_y) * stride",
            "subpixel_range": "0 <= subpixel_x, subpixel_y < binning",
            "binning_operation": "mean of synthesized observables, not mean atmosphere",
        },
    }
