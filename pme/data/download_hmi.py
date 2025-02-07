import argparse
import os

import drms
from dateutil.parser import parse


def donwload_ds(ds, dir, client, process=None):
    os.makedirs(dir, exist_ok=True)
    r = client.export(ds, protocol='fits', process=process)
    r.wait()
    download_result = r.download(dir)
    return download_result


def download_series(download_dir, email, t_start, t_end=None,
                    cadence='720s', series='S_720s', segments=None):
    os.makedirs(download_dir, exist_ok=True)
    client = drms.Client(email=email)

    if t_end is None:
        ds = f'hmi.{series}[{t_start.isoformat("_", timespec="seconds")}]'
    else:
        duration = (t_end - t_start).total_seconds()
        ds = f'hmi.{series}[{t_start.isoformat("_", timespec="seconds")}/{duration}s@{cadence}]'
    if segments is not None:
        ds += f'{{{segments}}}'
    return donwload_ds(ds, download_dir, client)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_dir', type=str, required=True)
    parser.add_argument('--email', type=str, required=True)
    parser.add_argument('--t_start', type=str, required=True)
    parser.add_argument('--t_end', type=str, required=False, default=None)
    parser.add_argument('--cadence', type=str, required=False, default='720s')
    parser.add_argument('--series', type=str, required=False, default='S_720s')
    args = parser.parse_args()

    t_start = args.t_start
    t_end = args.t_end

    series = args.series
    cadence = args.cadence

    download_dir = args.download_dir
    email = args.email

    t_start = parse(t_start)
    t_end = parse(t_end) if t_end is not None else None

    download_series(download_dir=download_dir, email=email,
                    t_start=t_start, t_end=t_end,
                    cadence=cadence, series=series)


if __name__ == '__main__':
    main()


def main():  # workaround for entry_points
    pass
