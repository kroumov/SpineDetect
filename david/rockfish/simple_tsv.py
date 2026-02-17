import json
from math import ceil
from pathlib import Path
from argparse import ArgumentParser

if __name__ == "__main__":

    ap = ArgumentParser()
    ap.add_argument('--path')
    ap.add_argument('x_nm', type=int)
    ap.add_argument('y_nm', type=int)
    ap.add_argument('z_nm', type=int)

    args = ap.parse_args()

    volume = None
    with open(Path(args.path)) as f:
        volume = json.loads(f.read())

    x_nm = args.x_nm
    y_nm = args.y_nm
    z_nm = args.z_nm

    x_svx = volume['out_svx_res']['x']
    y_svx = volume['out_svx_res']['y']
    z_svx = volume['out_svx_res']['z']

    x_n = ceil(x_nm / x_svx)
    y_n = ceil(y_nm / y_svx)
    z_n = ceil(z_nm / z_svx)

    output_path = Path(args.path).parent.joinpath('jobs.csv')
    with open(output_path, 'w') as output:
        for xx in range(x_n):
            for yy in range(y_n):
                for zz in range(z_n):
                    output.write(f'{xx},{yy},{zz}\n')