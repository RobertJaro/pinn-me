import os

from sunpy.map import Map


def load_hmi_date_obs_map(path):
    """Load an HMI map whose entire coordinate frame is referenced to DATE-OBS."""
    source_map = Map(path)
    date_obs = source_map.meta.get('DATE-OBS')
    if not date_obs:
        raise ValueError(f'HMI segment {os.fspath(path)!r} is missing the DATE-OBS keyword.')

    # HMIMap normally overrides the WCS reference time with T_OBS. Remove that
    # override, and DATE-AVG if present, so GenericMap/HMIMap consistently falls
    # back to DATE-OBS for the map frame, observer, and coordinate transforms.
    date_obs_meta = source_map.meta.copy()
    date_obs_meta.pop('T_OBS', None)
    date_obs_meta.pop('DATE-AVG', None)
    return Map(source_map.data, date_obs_meta)
