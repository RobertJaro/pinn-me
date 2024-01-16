import argparse
import os

import drms
from dateutil.parser import parse

from pme.data.download import donwload_ds

parser = argparse.ArgumentParser()
parser.add_argument('--download_dir', type=str, required=True)
parser.add_argument('--email', type=str, required=True)
parser.add_argument('--t_start', type=str, required=True)
parser.add_argument('--t_end', type=str, required=False, default=None)
args = parser.parse_args()

time = parse(args.t_start)
t_end = parse(args.t_end) if args.t_end is not None else None

os.makedirs(args.download_dir, exist_ok=True)
client = drms.Client(email=(args.email), verbose=True)

if t_end is None:
    ds = f"hmi.S_720s[{time.isoformat('_', timespec='seconds')}]"
else:
    ds = f"hmi.S_720s[{time.isoformat('_', timespec='seconds').replace('-', '.')}-{t_end.isoformat('_', timespec='seconds').replace('-', '.')}]"
donwload_ds(ds, args.download_dir, client)
