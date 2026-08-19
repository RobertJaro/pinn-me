import glob
import os.path
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames

from pme.data.hmi_time import load_hmi_date_obs_map


def _carrington_coordinate_grids(s_map, center_lon):
    """Return per-pixel Carrington longitude and latitude grids in degrees."""
    height, width = s_map.data.shape
    pixel_x, pixel_y = np.meshgrid(np.arange(width), np.arange(height))
    map_coords = s_map.pixel_to_world(pixel_x * u.pix, pixel_y * u.pix)
    carrington_frame = frames.HeliographicCarrington(
        observer=s_map.observer_coordinate,
        obstime=s_map.date,
    )
    carrington_coords = map_coords.transform_to(carrington_frame)

    # Keep longitude continuous across the 0/360-degree boundary while
    # retaining Carrington longitude values centered on the requested cutout.
    wrap_angle = u.Quantity(center_lon, u.deg) + 180 * u.deg
    longitude = carrington_coords.lon.wrap_at(wrap_angle).to_value(u.deg)
    latitude = carrington_coords.lat.to_value(u.deg)
    return longitude, latitude


class _Converter:
    """
    Convert full-disk HMI maps to subframes defined by:
      - center_lon: astropy Quantity (angle, e.g. 120 * u.deg) or plain number (deg assumed)
      - center_lat: astropy Quantity (angle, e.g. 10 * u.deg) or plain number (deg assumed)
      - shape_px: (width_pixels, height_pixels) as ints
      - pixel_scale: astropy Quantity (e.g. 0.5 * u.arcsec) or plain number (arcsec assumed)

    The angular extent is computed from shape_px * pixel_scale and the submap is
    selected in the Carrington frame using the map's observer.
    """

    def __init__(self, center_lon, center_lat, size, out_path, overwrite=False):
        # Coerce longitude/latitude to astropy Quantity in degrees

        self.center_lon = center_lon
        self.center_lat = center_lat
        # size is (width_pixels, height_pixels)
        self.size = size
        self.out_path = out_path
        self.overwrite = overwrite

    def convert(self, file):
        save_path = f"{self.out_path}/{os.path.basename(file)}"
        if os.path.exists(save_path) and not self.overwrite:
            print(f"Skipping {file}, already exists at {save_path}")
            return

        source_map = load_hmi_date_obs_map(file)
        s_map = source_map

        carr_center = SkyCoord(
            lon=self.center_lon,
            lat=self.center_lat,
            frame=frames.HeliographicCarrington,
            observer=s_map.observer_coordinate,
            obstime=s_map.date,
        )

        hp_center = carr_center.transform_to(s_map.coordinate_frame)

        # 3) Desired cutout size in *pixels*
        nx, ny = self.size  # e.g., (200, 200)

        # Convert pixel size -> world size using map scale (e.g., arcsec/pix)
        width = s_map.scale[0] * nx  # quantity, arcsec
        height = s_map.scale[1] * ny  # quantity, arcsec

        # 4) World-space corners around the transformed center
        bottom_left = SkyCoord(
            hp_center.Tx - width / 2,
            hp_center.Ty - height / 2,
            frame=hp_center.frame
        )
        top_right = SkyCoord(
            hp_center.Tx + width / 2,
            hp_center.Ty + height / 2,
            frame=hp_center.frame
        )

        # Extract submap
        s_map = s_map.submap(bottom_left=bottom_left, top_right=top_right)
        # Preserve the exact origin of the cutout on the physical HMI CCD so
        # the spatial transmission calibration can be cropped during loading.
        s_map.meta['CCD_X0'] = float(source_map.meta['CRPIX1'] - s_map.meta['CRPIX1'])
        s_map.meta['CCD_Y0'] = float(source_map.meta['CRPIX2'] - s_map.meta['CRPIX2'])
        s_map.meta['CCD_NX'] = int(source_map.data.shape[1])
        s_map.meta['CCD_NY'] = int(source_map.data.shape[0])
        s_map.save(save_path, overwrite=True)

        carrington_lon, carrington_lat = _carrington_coordinate_grids(
            s_map, self.center_lon
        )
        fig, axes = plt.subplots(
            3, 1, figsize=(12, 14), sharex=True, sharey=True,
            constrained_layout=True,
        )

        signal_image = axes[0].imshow(s_map.data, cmap='gray', origin='lower')
        signal_unit = s_map.meta.get('BUNIT')
        signal_label = f'HMI signal [{signal_unit}]' if signal_unit else 'HMI signal'
        signal_cax = make_axes_locatable(axes[0]).append_axes(
            'right', size='3%', pad=0.08
        )
        fig.colorbar(signal_image, cax=signal_cax, label=signal_label)
        axes[0].set_title('HMI subframe')

        longitude_image = axes[1].imshow(
            carrington_lon, cmap='twilight', origin='lower'
        )
        longitude_cax = make_axes_locatable(axes[1]).append_axes(
            'right', size='3%', pad=0.08
        )
        fig.colorbar(
            longitude_image, cax=longitude_cax,
            label='Carrington longitude [deg]'
        )
        axes[1].set_title('Carrington longitude')

        latitude_image = axes[2].imshow(
            carrington_lat, cmap='viridis', origin='lower'
        )
        latitude_cax = make_axes_locatable(axes[2]).append_axes(
            'right', size='3%', pad=0.08
        )
        fig.colorbar(
            latitude_image, cax=latitude_cax,
            label='Carrington latitude [deg]'
        )
        axes[2].set_title('Carrington latitude')

        for ax in axes:
            ax.set_xlabel('Image x [pixel]')
            ax.set_ylabel('Image y [pixel]')

        fig.suptitle(f"Submap from {os.path.basename(file)}")
        fig.savefig(f"{self.out_path}/{os.path.basename(file).replace('.fits', '.jpg')}", dpi=300)
        plt.close(fig)
        print(f"Converted and saved {file} to {save_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Subframe HMI converter')
    parser.add_argument('--input_files', required=True, help='glob pattern for HMI files')
    parser.add_argument('--out_path', required=True, help='output directory')
    parser.add_argument('--longitude', type=float, required=True, help='center longitude in degrees')
    parser.add_argument('--latitude', type=float, required=True, help='center latitude in degrees')
    parser.add_argument('--size', type=int, nargs=2, required=True,
                        help='size of the subframe in pixels (width height)')
    parser.add_argument('--processes', type=int, default=16, help='number of worker processes')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing files')
    args = parser.parse_args()

    hmi_files = sorted(glob.glob(args.input_files))
    out_path = args.out_path
    longitude = args.longitude * u.deg
    latitude = args.latitude * u.deg
    size = (args.size[0], args.size[1]) * u.pix
    processes = args.processes

    os.makedirs(out_path, exist_ok=True)

    with Pool(processes=processes) as pool:
        converter = _Converter(center_lon=longitude, center_lat=latitude,
                               size=size, out_path=out_path,
                               overwrite=args.overwrite)
        pool.map(converter.convert, hmi_files)
