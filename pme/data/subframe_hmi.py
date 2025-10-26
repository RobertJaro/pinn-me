import glob
import os.path
from multiprocessing import Pool

import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.map import Map


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

        s_map = Map(file)

        carr_center = SkyCoord(lon=self.center_lon, lat=self.center_lat, frame=frames.HeliographicCarrington, observer=s_map.observer_coordinate)

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
        s_map.save(save_path, overwrite=True)

        fig, ax = plt.subplots(figsize=(10, 10))

        im = ax.imshow(s_map.data, cmap='gray', origin='lower')
        plt.colorbar(im, ax=ax, orientation='vertical', label='Intensity')

        ax.set_title(f"Submap from {os.path.basename(file)}")
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
