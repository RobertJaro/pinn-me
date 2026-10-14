"""Spatial selection for additional, unmodified polarized Stokes profiles."""
import numpy as np
from scipy.ndimage import gaussian_filter


def polarization_selection(stokes, radius, valid=None, smoothing_sigma=5,
                           fraction=0.1, radius_limit=0.95, continuum_quantile=0.9):
    """Return a mask and diagnostics; all lengths are in input-image pixels.

    P = sqrt(mean_lambda(Q**2 + U**2 + V**2)) / quantile_lambda(I).
    Invalid/outside-radius pixels contribute neither values nor kernel weight.
    Percentile ties are excluded, so a constant map selects no pixels.
    """
    if not np.isfinite(smoothing_sigma) or smoothing_sigma <= 0:
        raise ValueError('smoothing_sigma must be finite and positive.')
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('fraction must be between zero and one.')
    if not np.isfinite(radius_limit) or not 0 < radius_limit <= 1:
        raise ValueError('radius_limit must be in (0, 1].')
    if not np.isfinite(continuum_quantile) or not 0 < continuum_quantile <= 1:
        raise ValueError('continuum_quantile must be in (0, 1].')
    stokes = np.asarray(stokes)
    if stokes.ndim != 4 or stokes.shape[-2] != 4 or radius.shape != stokes.shape[:2]:
        raise ValueError('Expected Stokes [height, width, 4, wavelength] and matching radius.')
    continuum = np.quantile(stokes[..., 0, :], continuum_quantile, axis=-1)
    eligible = np.isfinite(stokes).all(axis=(-2, -1)) & (continuum > 0)
    eligible &= np.isfinite(radius) & (radius < radius_limit)
    if valid is not None:
        eligible &= valid
    # Accumulate without allocating a second full Q/U/V cube.
    power = np.zeros(stokes.shape[:2], dtype=np.float64)
    for component in range(1, 4):
        for wavelength in range(stokes.shape[-1]):
            values = stokes[..., component, wavelength].astype(np.float64)
            power += values * values / stokes.shape[-1]
    with np.errstate(invalid='ignore', divide='ignore'):
        score = np.sqrt(power) / continuum
    eligible &= np.isfinite(score)
    if not eligible.any():
        raise ValueError('Polarization selection has no eligible pixels.')
    score[~eligible] = np.nan
    weights = gaussian_filter(eligible.astype(float), smoothing_sigma, mode='constant', cval=0)
    numerator = gaussian_filter(np.where(eligible, score, 0), smoothing_sigma, mode='constant', cval=0)
    smoothed = np.full(score.shape, np.nan)
    np.divide(numerator, weights, out=smoothed, where=eligible & (weights > 0))
    threshold = float(np.quantile(smoothed[eligible], 1 - fraction))
    mask = eligible & (smoothed > threshold)
    if not mask.any():
        raise ValueError('Polarization selection is empty (possibly a constant polarization map).')
    return mask, smoothed, {
        'threshold': threshold, 'eligible_pixels': int(eligible.sum()),
        'selected_pixels': int(mask.sum()), 'smoothing_sigma': smoothing_sigma,
        'fraction': fraction, 'radius_limit': radius_limit,
        'continuum_quantile': continuum_quantile,
    }
