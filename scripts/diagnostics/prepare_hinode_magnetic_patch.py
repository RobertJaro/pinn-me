"""Make an 8x8 diagnostic crop around the strongest observed polarization.

Run from the repository root. No magnetic truth or target field is used.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from prom3theus.observations.store import ObservationStore
from prom3theus.observations.contracts import ObservationRaster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/hinode_single_frame'))
    parser.add_argument('--output', type=Path, default=Path('data/local_kg_patch/observations'))
    args = parser.parse_args()
    raster = ObservationStore.load(args.source, mmap=False)
    stokes = torch.as_tensor(raster.stokes)
    indices = raster.metadata['normalization']['indices']
    continuum = stokes[..., 0, indices].mean(-1).clamp_min(.1)
    strength = (stokes[..., 1:, :] / continuum[..., None, None]).square().mean((-1, -2))
    strength = strength.masked_fill(~torch.as_tensor(raster.valid_mask), -torch.inf)
    index = int(strength.argmax())
    y, x = divmod(index, stokes.shape[1])
    y0 = min(max(y-4, 0), stokes.shape[0]-8)
    x0 = min(max(x-4, 0), stokes.shape[1]-8)
    region = (slice(y0, y0+8), slice(x0, x0+8))
    arrays = {key: torch.as_tensor(getattr(raster, key)[region]).clone()
              for key in ObservationStore._CORE_FIELDS if key != 'wavelength_angstrom'}
    metadata = copy.deepcopy(dict(raster.metadata))
    xy = arrays['coordinates'][..., :2].reshape(-1, 2)
    lo, hi = xy.min(0).values, xy.max(0).values
    pixel_width = (hi-lo)/7
    metadata['coordinates']['network_affine'] = {
        'center_megameter': ((lo+hi)/2).tolist(),
        'scale_megameter': ((hi-lo+pixel_width)/2).tolist(),
    }
    selection = {'source': str(args.source), 'rows': [y0, y0+8],
                 'columns': [x0, x0+8], 'center_pixel_index': index,
                 'selection': 'largest observed continuum-normalized Q/U/V power'}
    metadata['local_validation_subset'] = selection
    patch = ObservationRaster(
        **arrays, wavelength_angstrom=torch.as_tensor(raster.wavelength_angstrom).clone(),
        metadata=metadata,
        auxiliary={key: torch.as_tensor(value[region]).clone()
                   for key, value in raster.auxiliary.items()},
    )
    signature = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
    ObservationStore.save_sequence(args.output, [patch], source_signature=signature,
                                   metadata={'source': 'local magnetic-recovery validation crop'})
    print(json.dumps(selection, indent=2))


if __name__ == '__main__':
    main()
