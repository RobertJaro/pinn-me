import glob
import os.path
from multiprocessing import Pool

import matplotlib.pyplot as plt
from astropy.io import fits
from sunpy.coordinates import frames
from sunpy.map import Map


class _Converter:

    def __init__(self, bottom_left, top_right, out_path, overwrite=False):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.out_path = out_path
        self.overwrite = overwrite

    def convert(self, file):
        save_path = f"{self.out_path}/{os.path.basename(file)}"
        if os.path.exists(save_path) and not self.overwrite:
            print(f"Skipping {file}, already exists at {save_path}")
            return

        s_map = Map(file)
        s_map = s_map.submap(bottom_left=self.bottom_left, top_right=self.top_right)
        s_map.save(save_path, overwrite=True)

        fig, ax = plt.subplots(figsize=(10, 10))

        im = ax.imshow(s_map.data, cmap='gray', origin='lower')
        plt.colorbar(im, ax=ax, orientation='vertical', label='Intensity')

        ax.set_title(f"Submap from {os.path.basename(file)}")
        fig.savefig(f"{self.out_path}/{os.path.basename(file).replace('.fits', '.jpg')}", dpi=300)
        plt.close(fig)
        print(f"Converted and saved {file} to {save_path}")


if __name__ == '__main__':
    hmi_files = sorted(glob.glob('/glade/work/rjarolim/data/hmi_stokes/20240323_720s/*.fits'))
    ref_file = '/glade/work/rjarolim/data/phi_stokes/LB5/solo_L2_phi-hrt-stokes_20240323T223009_V01_0443230201.fits'
    out_path = '/glade/work/rjarolim/data/phi_stokes/LB5_hmi'

    os.makedirs(out_path, exist_ok=True)

    header = fits.getheader(ref_file)
    data = fits.getdata(ref_file)
    ref_map = Map(data, header)

    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(ref_map.data, cmap='gray', origin='lower')
    plt.colorbar(im, ax=ax, orientation='vertical', label='Intensity')

    ax.set_title("Reference Map")
    fig.savefig(f'{out_path}/reference_map.jpg', dpi=300)

    bottom_left = ref_map.bottom_left_coord.transform_to(frames.HeliographicCarrington)
    top_right = ref_map.top_right_coord.transform_to(frames.HeliographicCarrington)

    with Pool(processes=8) as pool:
        converter = _Converter(bottom_left, top_right, out_path)
        pool.map(converter.convert, hmi_files)
