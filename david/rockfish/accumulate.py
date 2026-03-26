from argparse import ArgumentParser
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt

import tifffile as tf
from tqdm import tqdm

def accumulate(cache_path):
    regex = r"^svx_(\d+)_(\d+)_(\d+)\.npy$"

    svx_coords = []

    for f in cache_path.iterdir():
        m = re.match(regex, f.name)
        if m is None: continue
        svx_coord = np.array(m.groups()).astype(int)
        svx_coords.append(svx_coord)

    svx_coords = np.array(svx_coords)
    dims = np.max(svx_coords, axis=0) - np.min(svx_coords, axis=0)
    dims = dims + np.ones_like(dims)

    first_file = next(cache_path.iterdir())
    first_frag = np.load(first_file)
    shape = first_frag.shape

    acc = np.empty(shape * dims, dtype=first_frag.dtype)

    for f in tqdm(cache_path.iterdir(), total=len([_ for _ in cache_path.iterdir()])):
        m = re.match(regex, f.name)
        if m is None:
            print(f'{ f.name } - DID NOT MATCH')
            continue
        svx_coord = np.array(m.groups()).astype(int)
        frag = np.load(f)

        assert frag.shape == shape

        a = svx_coord * shape
        b = a + shape

        x0, y0, z0 = a
        x1, y1, z1 = b

        acc[x0:x1, y0:y1, z0:z1] = frag
        
    np.save(cache_path.joinpath('../svx_acc.npy'))


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--path')

    args = ap.parse_args()

    cache_path = Path(args.path)
    assert cache_path.exists()
    
    accumulate(cache_path.joinpath('output/svx'))