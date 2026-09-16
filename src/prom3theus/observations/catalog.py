"""Bounded canonical validation and disposable, versioned row-count sidecars."""

from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from .bulk import PixelCatalog
from .loading import _restore_raster, _plain


def validated_raster(cls, *, budget_bytes=32 * 1024**2, **values):
    """Run the canonical constructor on owned slabs, retaining original maps.

    Empty slabs include a copied valid anchor row so the constructor can check
    geometry everywhere without imposing valid-pixel checks on masked payloads.
    """
    mask = values["valid_mask"]
    if mask.ndim != 2 or mask.dtype != torch.bool or not mask.numel():
        return cls(**values)  # canonical structural error
    arrays = {
        name: value
        for name, value in values.items()
        if isinstance(value, torch.Tensor) and name != "wavelength_angstrom"
    }
    arrays.update(
        {
            "auxiliary:" + key: value
            for key, value in values.get("auxiliary", {}).items()
        }
    )
    if any(tuple(value.shape[:2]) != tuple(mask.shape) for value in arrays.values()):
        return cls(**values)
    row_bytes = sum(
        value[0:1].numel() * value.element_size() for value in arrays.values()
    )
    rows = max(1, budget_bytes // max(row_bytes, 1))
    anchor = None
    for start in range(0, mask.shape[0], rows):
        valid = mask[start : start + rows].clone()
        if valid.any():
            anchor = start + int(torch.nonzero(valid.any(dim=1))[0])
            break
    if anchor is None:
        raise ValueError("An observation raster requires at least one valid pixel")
    normalized = None
    for start in range(0, mask.shape[0], rows):
        slabs = {
            key: value[start : start + rows].clone(
                memory_format=torch.contiguous_format
            )
            for key, value in arrays.items()
        }
        if not slabs["valid_mask"].any():
            slabs = {
                key: torch.cat((value, arrays[key][anchor : anchor + 1].clone()))
                for key, value in slabs.items()
            }
        options = {
            **values,
            **{
                key: value
                for key, value in slabs.items()
                if not key.startswith("auxiliary:")
            },
        }
        if "auxiliary" in values:
            options["auxiliary"] = {
                key.removeprefix("auxiliary:"): value
                for key, value in slabs.items()
                if key.startswith("auxiliary:")
            }
        checked = cls(**options)
        if normalized is None:
            normalized = {
                field.name: getattr(checked, field.name) for field in fields(checked)
            }
        # Only retain normalized scalar/metadata values, never slab tensors.
        for key in arrays:
            if not key.startswith("auxiliary:"):
                normalized[key] = values[key]
        if "auxiliary" in values:
            from types import MappingProxyType

            normalized["auxiliary"] = MappingProxyType(values["auxiliary"])
        del checked, slabs, options
    return _restore_raster(
        cls, {key: _plain(value) for key, value in normalized.items()}
    )


def attach_catalog(raster, root, record):
    """Cache small prefix counts against the validated mask and file identity.

    This does not replace payload checksum or scientific validation. A stale or
    unwritable sidecar falls back to a sequential mask scan.
    """
    root = Path(root)
    mask_record = record["arrays"]["valid_mask"]
    path = root / mask_record["file"]
    stat = path.stat()
    identity = {
        "version": PixelCatalog.version,
        "rows": min(256, max(1, 65536 // raster.valid_mask.shape[1])),
        "record": mask_record,
        "file_state": [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino],
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    sidecar = root / ("bulk-catalog-" + key + ".json")
    catalog = None
    try:
        data = json.loads(sidecar.read_text())
        prefix = data["prefix"]
        row_prefix = data["row_prefix"]
        rows = identity["rows"]
        height, width = raster.valid_mask.shape
        if (
            data["identity"] == identity
            and len(row_prefix) == height + 1
            and row_prefix[0] == 0
            and all(type(value) is int for value in row_prefix)
            and all(0 <= b - a <= width for a, b in zip(row_prefix, row_prefix[1:]))
            and prefix
            == row_prefix[::rows] + ([row_prefix[-1]] if height % rows else [])
            and len(prefix) == (height + rows - 1) // rows + 1
            and prefix[0] == 0
            and all(type(x) is int for x in prefix)
            and all(
                0 <= b - a <= min(rows, height - i * rows) * width
                for i, (a, b) in enumerate(zip(prefix, prefix[1:]))
            )
        ):
            catalog = PixelCatalog.__new__(PixelCatalog)
            catalog.mask, catalog.rows, catalog.prefix = raster.valid_mask, rows, prefix
            catalog.row_prefix = row_prefix
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if catalog is None:
        catalog = PixelCatalog(raster.valid_mask)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=root, prefix=".bulk-", delete=False
            ) as handle:
                temporary = handle.name
                json.dump(
                    {
                        "identity": identity,
                        "prefix": catalog.prefix,
                        "row_prefix": catalog.row_prefix,
                    },
                    handle,
                )
            os.replace(temporary, sidecar)
        except OSError:
            pass
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
    object.__setattr__(raster, "_bulk_catalog", catalog)
