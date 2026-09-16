"""Draw a subframe on an AIA image and print Carrington settings for prepare.sh."""

import argparse
from pathlib import Path

import astropy.units as u
import numpy as np
from sunpy.coordinates import frames
from prom3theus.observations.pixel_footprint import HMI_PIXEL_SCALE_ARCSEC, pixel_dimensions


def selection_parameters(image_map, bounds):
    """Report the selected size in nominal HMI pixels, without a reference file."""
    left, right, bottom, top = bounds
    x0, y0 = np.floor(np.array([left, bottom]) + 0.5).astype(int)
    x1, y1 = np.ceil(np.array([right, top]) + 0.5).astype(int)
    height, width = image_map.data.shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Draw a rectangle inside the image.")
    frame = frames.HeliographicCarrington(
        observer=image_map.observer_coordinate, obstime=image_map.reference_date,
    )
    center = image_map.pixel_to_world((x0 + x1 - 1) / 2 * u.pix, (y0 + y1 - 1) / 2 * u.pix)
    center = center.transform_to(frame)
    xs = np.linspace(x0 - 0.5, x1 - 0.5, 33)
    ys = np.linspace(y0 - 0.5, y1 - 0.5, 33)
    edge = image_map.pixel_to_world(
        np.concatenate([xs, xs, np.full(33, xs[0]), np.full(33, xs[-1])]) * u.pix,
        np.concatenate([np.full(33, ys[0]), np.full(33, ys[-1]), ys, ys]) * u.pix,
    ).transform_to(frame)
    longitude = (edge.lon - center.lon).wrap_at(180 * u.deg).to_value(u.deg)
    latitude = (edge.lat - center.lat).to_value(u.deg)
    if not np.isfinite(np.concatenate([longitude, latitude])).all():
        raise ValueError("Keep the entire rectangle on the visible solar disk.")
    width_pixels, height_pixels = pixel_dimensions(
        x1 - x0, y1 - y0, u.Quantity(image_map.scale).to_value(u.arcsec / u.pix),
        HMI_PIXEL_SCALE_ARCSEC,
    )
    if max(width_pixels, height_pixels) > 4096:
        raise ValueError("Selection exceeds the HMI detector size; select a smaller region.")
    return {
        "longitude_deg": float(center.lon.to_value(u.deg) % 360),
        "latitude_deg": float(center.lat.to_value(u.deg)),
        "width_pixels": width_pixels,
        "height_pixels": height_pixels,
    }


def shell_settings(parameters):
    return "\n".join(f"{key}={value if isinstance(value, int) else f'{value:.6f}'}"
                     for key, value in parameters.items())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="full-disk AIA FITS image")
    args = parser.parse_args(argv)

    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector
    from sunpy.map import Map
    from astropy.visualization import AsinhStretch, ImageNormalize

    source = Map(str(args.image))
    if "AIA" not in source.instrument.upper():
        parser.error("Select a full-disk AIA FITS image.")
    height, width = source.data.shape
    stride = max(1, int(np.ceil(max(height, width) / 1024)))
    display = source.data[::stride, ::stride]
    finite = display[np.isfinite(display)]
    if not finite.size:
        parser.error("The image contains no finite pixels.")
    figure, axes = plt.subplots(figsize=(10, 10))
    figure.subplots_adjust(bottom=0.24)
    norm = ImageNormalize(vmin=np.percentile(finite, 1), vmax=np.percentile(finite, 99.5),
                          stretch=AsinhStretch(), clip=True)
    # Stretch/colorize the small preview once, not all 16M pixels on every drag.
    rgba = plt.get_cmap(source.plot_settings["cmap"])(norm(display), bytes=True)
    ny, nx = display.shape
    axes.imshow(rgba, origin="lower", interpolation="nearest",
                extent=(-stride / 2, (nx - 0.5) * stride,
                        -stride / 2, (ny - 0.5) * stride))
    # Preview sample centers remain at original detector pixels 0, stride, ... .
    axes.set_xlim(-0.5, width - 0.5)
    axes.set_ylim(-0.5, height - 0.5)
    axes.set(title="Drag a rectangle; resize using its handles. Enter: accept. Esc: cancel.",
             xlabel="AIA detector x [pixels]", ylabel="AIA detector y [pixels]")
    status = figure.text(0.1, 0.03, "Select an on-disk region. Output sizes are HMI pixels.",
                         family="monospace")
    selected = {}

    def update(*_):
        selected.clear()
        try:
            selected.update(selection_parameters(source, selector.extents))
            status.set_text(shell_settings(selected))
        except ValueError as error:
            status.set_text(str(error))
        figure.canvas.draw_idle()

    selector = RectangleSelector(axes, update, button=[1], interactive=True,
                                 useblit=figure.canvas.supports_blit,
                                 minspanx=2, minspany=2, spancoords="data")

    def keypress(event):
        if event.key == "enter" and selected:
            print(shell_settings(selected))
            plt.close(figure)
        elif event.key == "escape":
            plt.close(figure)

    figure.canvas.mpl_connect("key_press_event", keypress)
    plt.show()


if __name__ == "__main__":
    main()
