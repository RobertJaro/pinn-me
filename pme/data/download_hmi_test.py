import argparse

from dateutil.parser import parse

from pme.data.download_hmi import download_series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_dir', type=str, required=True)
    parser.add_argument('--email', type=str, required=True)
    parser.add_argument('--t_start', type=str, required=True)
    args = parser.parse_args()

    t_start = args.t_start

    download_dir = args.download_dir
    email = args.email

    t_start = parse(t_start)

    download_series(download_dir=download_dir, email=email, t_start=t_start, series='B_720s',
                    segments='inclination, azimuth, field, disambig')


if __name__ == '__main__':
    main()