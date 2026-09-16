"""Publish validated downloads without destroying the previous dataset."""

import os
from pathlib import Path
import tempfile


def download_fits_export(request, directory: Path):
    """Use DRMS filenames, rejecting malformed exports before any transfer."""
    request.wait()
    for record, filename in zip(request.urls["record"], request.urls["filename"], strict=True):
        if (
            not isinstance(filename, str)
            or not filename.strip()
            or filename in {".", ".."}
            or Path(filename).name != filename
        ):
            raise RuntimeError(
                f"JSOC export {request.id!r} returned an invalid FITS filename "
                f"{filename!r} for record {record!r} "
                f"(method={request.method!r}, protocol={request.protocol!r}). "
                "No files downloaded; inspect the export response."
            )
    return request.download(str(directory), fname_from_rec=False)


def publish_download(staging: Path, output: Path, *, overwrite: bool) -> None:
    """Replace only after validation; retain an existing dataset as a backup."""
    backup = None
    if os.path.lexists(output):
        if not overwrite:
            raise FileExistsError(f"Download output already exists: {output}.")
        backup = Path(tempfile.mkdtemp(prefix=f".{output.name}.previous-", dir=output.parent)) / output.name
        os.replace(output, backup)
    try:
        os.replace(staging, output)
    except BaseException:
        if backup is not None:
            os.replace(backup, output)
        raise
