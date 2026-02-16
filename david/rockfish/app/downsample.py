import json
import sys
from math import lcm
from pathlib import Path
import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from caveclient import CAVEclient
from cloudvolume import CloudVolume, Bbox, Vec

from get_all_cts import get_all_cts


def weights_for_1d_voxel_downscaling(
        p: int,
        dx_in,
        dx_out
    ):

    assert dx_in > 0 and dx_out > 0

    # Output voxel interval
    x0 = p * dx_out
    x1 = x0 + dx_out

    i_start = int(np.floor(x0 / dx_in))
    i_end = int(np.ceil(x1 / dx_in))
    idx = np.arange(i_start, i_end + 1, dtype=np.int64)

    in_left = idx * dx_in
    in_right = in_left + dx_in

    overlap = np.minimum(in_right, x1) - np.maximum(in_left, x0)
    overlap = np.maximum(overlap, 0.0)

    idx = idx[overlap > 1e-12]
    overlap = overlap[overlap > 1e-12]

    w = overlap / dx_in

    return idx, w


def resample(V, vx_coord, din, dout):
    Nx, Ny, Nz = V.shape

    i, j ,k = vx_coord
    dx_in, dy_in, dz_in = din
    dx_out, dy_out, dz_out = dout

    idx_x, wx = weights_for_1d_voxel_downscaling(i, dx_in, dx_out)
    idx_y, wy = weights_for_1d_voxel_downscaling(i, dy_in, dy_out)
    idx_z, wz = weights_for_1d_voxel_downscaling(i, dz_in, dz_out)

    block = V[np.ix_(idx_x, idx_y, idx_z)]
    W = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
    val = float(np.sum(block * W))

    return val


def range_helper(p, dx_in, dx_out):
    x0 = p * dx_out
    x1 = x0 + dx_out

    i_start = int(np.floor(x0 / dx_in))
    i_end = int(np.ceil(x1 / dx_in))
    idx = np.arange(i_start, i_end, dtype=np.int64)
    
    # return (i_start, i_end)
    return idx


def compute_new_voxel(seg, seg_res, vx_coord, vx_res, cells, method):
    
    idx_x = range_helper(vx_coord[0], seg_res[0], vx_res[0])
    idx_y = range_helper(vx_coord[1], seg_res[1], vx_res[1])
    idx_z = range_helper(vx_coord[2], seg_res[2], vx_res[2])

    block = seg[np.ix_(idx_x, idx_y, idx_z)]

    unique_ids = np.unique(block)

    if method == 'occupancy':
        unique_ids = set(unique_ids)
        cells = set(cells)

        if len(unique_ids & cells) != 0:
            return 1
        else:
            return 0
        
    elif method == 'density':
        val = 0
        for uid in unique_ids:
            if uid not in cells: continue

            neuron_image = np.zeros(block.shape[:-1])
            mask = np.any(block == uid, axis=-1)
            neuron_image[mask] = 1

            val += resample(neuron_image, (0, 0, 0), seg_res, vx_res)

    return val


def compute_supervoxel(seg, seg_res, seg_origin, svx_coord, svx_size, vx_res, cells, method):
    a = seg_origin + Vec(
        svx_size[0] * svx_coord[0],
        svx_size[1] * svx_coord[1],
        svx_size[2] * svx_coord[2]
    )
    b = a + Vec(
        svx_size[0],
        svx_size[1],
        svx_size[2]
    )
    cut = Bbox(a, b) / seg_res
    cut = cut.astype(int)

    block = seg.image.download(cut, mip=0)

    nx = int(svx_size[0] / vx_res[0])
    ny = int(svx_size[1] / vx_res[1])
    nz = int(svx_size[2] / vx_res[2])

    svx = np.zeros((nx, ny, nz))

    for idx, _ in tqdm(np.ndenumerate(svx), total=svx.size):
        svx[idx] = compute_new_voxel(block, seg_res, idx, vx_res, cells, method)

    return svx


# TODO: Save other metadata (coordinate systems, sources, etc) to a JSON
# TODO: Compare existing JSON with current JSON before modifying store
def memoize_supervoxel(folder, seg, seg_res, seg_origin, svx_coord, svx_size, vx_res, cells, method, flush=False):
    filename = f'svx_{svx_coord[0]}_{svx_coord[1]}_{svx_coord[2]}'
    
    folder = Path(folder)
    if folder.exists() == False:
        os.makedirs(str(folder.absolute()))

    filepath = folder.joinpath(filename)
    if filepath.exists():
        if flush:
            os.remove(str(filepath.absolute()))
        else:
            return np.load(str(filepath.absolute()))
        
    svx = compute_supervoxel(seg, seg_res, seg_origin, svx_coord, svx_size, vx_res, cells, method)
    np.save(filepath, svx)

    return svx


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to folder to store the downloaded supervoxel')
    parser.add_argument('svx_x', type=int)
    parser.add_argument('svx_y', type=int)
    parser.add_argument('svx_z', type=int)
    parser.add_argument('-s', '--silent', action='store_true')

    args = parser.parse_args()

    config = None
    config_path = Path(args.path).joinpath("volume.json")
    with open(config_path, 'r') as file:
        config = json.loads(file.read())

    if config is None:
        print('Could not open the config file!')
        print('Make sure there is a `volume.json` in the output directory')
        print(config_path)
        sys.exit(1)
        
    dataset_name = config['dataset']
    materialization = config['materialization']
    
    client = CAVEclient(dataset_name)
    client.version = materialization

    seg_cv = CloudVolume('precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg_m1300', progress=False, use_https=True)
    img_cv = CloudVolume('precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/em', progress=True, use_https=True)

    meta = client.materialize.get_table_metadata("allen_v1_column_types_slanted_ref")
    ann_res = Vec(*meta["voxel_resolution"])
    seg_res = seg_cv.resolution

    cells = None
    if config.get('whitelist') is not None:
        cells = config['whitelist']
    else:
        ct_all_df = get_all_cts(client)
        cells = ct_all_df['pt_root_id'].values


    ###############################
    # NOTE: MUST BE EDGE ALIGNED!!!
    ###############################

    dx_in = int(seg_res[0])
    dx_out = config['out_voxel_res']['x']

    dy_in = int(seg_res[1])
    dy_out = config['out_voxel_res']['y']

    dz_in = int(seg_res[2])
    dz_out = config['out_voxel_res']['z']

    # Calculate the minimum number of original voxels to be edge aligned
    n_x = lcm(dx_in, dx_out) / dx_in    # (4, 100) -> 100 / 4 = 25
    n_y = lcm(dy_in, dy_out) / dy_in    # (4, 100) -> 100 / 4 = 25
    n_z = lcm(dz_in, dz_out) / dz_in    # (40, 100) -> 200 / 40 = 5

    svx_dx = config['out_svx_res']['x']
    svx_dy = config['out_svx_res']['y']
    svx_dz = config['out_svx_res']['z']

    assert (svx_dx % dx_in) == 0, f'svx_dx must be a multiple of the original resolution. try a multiple of {n_x}'
    assert (svx_dx % dx_out) == 0, f'svx_dx must be a multiple of the downsampled resolution. try a multiple of {n_x}'
    assert (svx_dy % dy_in) == 0, f'svy_dx must be a multiple of the original resolution. try a multiple of {n_y}'
    assert (svx_dy % dy_out) == 0, f'svy_dx must be a multiple of the downsampled resolution. try a multiple of {n_y}'
    assert (svx_dz % dz_in) == 0, f'svz_dx must be a multiple of the original resolution. try a multiple of {n_z}'
    assert (svx_dz % dz_out) == 0, f'svz_dx must be a multiple of the downsampled resolution. try a multiple of {n_z}'

    o = config['microns_origin']
    origin = Vec(o['x'], o['y'], o['z'])    # nm
    origin = origin.astype(int)             # nm

    folder = Path(args.folder).joinpath(config['output'])
    svx_coord = (args.svx_x, args.svx_y, args.svx_z)
    svx_size = (svx_dx, svx_dy, svx_dz)
    vx_res = (dx_out, dy_out, dz_out)
    method =  config['method']
    svx = memoize_supervoxel(folder, seg_cv, seg_res, origin, svx_coord, svx_size, vx_res, cells, method)

    if not args.silent:
        import matplotlib.pyplot as plt
        plt.imshow(svx[:,:,0], cmap='gray')
        plt.show()
