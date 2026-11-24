"""Small source references for reproducible native-atmosphere evaluation.

References contain orientation, geometry, and file metadata, never cube arrays.
File-stat verification detects ordinary replacement or edits without rescanning
large cubes; it is not a content checksum. Native z=0 is the bottom sample center,
not a measured photosphere or a registration to the inversion's solar radius.
"""

from __future__ import annotations

from pathlib import Path

from .reader import (
    MURaMSnapshot,
    NATIVE_REFERENCE_SCHEMA as SCHEMA,
    _SOURCE_FIELDS,
    native_snapshot_reference,
)


def verify_native_reference(reference, *, source_path=None):
    """Verify source file stats without opening cubes or reading their contents.

    Original files must retain path, size, mtime, and ctime. An explicitly
    relocated source directory must retain filenames, sizes, and mtimes; ctime
    changes when files are copied. Relocation should therefore preserve mtimes.
    """
    if reference.get("schema") != SCHEMA:
        raise ValueError("Unsupported MURaM native-reference schema.")
    original = Path(reference["source_path"]).resolve()
    source = original if source_path is None else Path(source_path).resolve()
    relocated = source != original
    if set(reference["files"]) != set(_SOURCE_FIELDS):
        raise ValueError("Incomplete native-reference source file list.")
    iteration = int(reference["iteration"])
    for name, field_id in _SOURCE_FIELDS.items():
        recorded = reference["files"][name]
        filename = recorded["name"]
        if filename not in {f"{field_id}.{iteration}", f"{field_id}.{iteration:06d}"}:
            raise ValueError(f"Invalid native-reference filename: {filename!r}.")
        path = source / filename
        # Match reader lookup precedence if a second, unpadded alias was added.
        preferred = source / f"{field_id}.{iteration}"
        if preferred.exists() and preferred.resolve() != path.resolve():
            raise ValueError(f"Native source file selection changed: {path}.")
        stat = path.stat()
        keys = ("size_bytes", "mtime_ns") if relocated else ("size_bytes", "mtime_ns", "ctime_ns")
        actual = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}
        if any(actual[key] != recorded[key] for key in keys) or (
            not relocated and str(path.resolve()) != recorded["path"]
        ):
            raise ValueError(f"Native source identity changed: {path}.")
    return source


def open_native_reference(reference, *, source_path=None, verify=True):
    """Reopen the original atmosphere lazily, optionally from a relocated copy.

    The returned snapshot uses native resolution. Apply the descriptor's
    pixel_mapping stride to its column source; binning describes averaging of
    observables and must not silently average atmosphere columns before transfer.
    Close the snapshot after evaluation to release its workers and mappings.
    """
    if reference.get("schema") != SCHEMA:
        raise ValueError("Unsupported MURaM native-reference schema.")
    source = (verify_native_reference(reference, source_path=source_path) if verify
              else Path(reference["source_path"] if source_path is None else source_path))
    reader = reference["reader"]
    return MURaMSnapshot(
        str(source), int(reference["iteration"]),
        axis_order=tuple(reader["axis_order"]), vertical_axis=reader["vertical_axis"],
        component_order=tuple(reader["component_order"]),
    )
