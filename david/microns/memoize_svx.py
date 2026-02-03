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


def compute_new_voxel(seg, seg_res, vx_coord, vx_res, ct_df):
    
    idx_x = range_helper(vx_coord[0], seg_res[0], vx_res[0])
    idx_y = range_helper(vx_coord[1], seg_res[1], vx_res[1])
    idx_z = range_helper(vx_coord[2], seg_res[2], vx_res[2])

    block = seg[np.ix_(idx_x, idx_y, idx_z)]

    unique_ids = np.unique(block)

    val = 0
    for uid in unique_ids:
        if uid not in ct_df['pt_root_id'].values: continue

        neuron_image = np.zeros(block.shape[:-1])
        mask = np.any(block == uid, axis=-1)
        neuron_image[mask] = 1

        val += resample(neuron_image, (0, 0, 0), seg_res, vx_res)

    return val


def compute_supervoxel(seg, seg_res, seg_origin, svx_coord, svx_size, vx_res, ct_df):
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
    print(svx.shape)

    for idx, _ in tqdm(np.ndenumerate(svx), total=svx.size):
        svx[idx] = compute_new_voxel(block, seg_res, idx, vx_res, ct_df)

    return svx


# TODO: Save other metadata (coordinate systems, sources, etc) to a JSON
# TODO: Compare existing JSON with current JSON before modifying store
def memoize_supervoxel(folder, seg, seg_res, seg_origin, svx_coord, svx_size, vx_res, ct_df, flush=False):
    filename = f'svx_{svx_coord[0]}_{svx_coord[1]}_{svx_coord[2]}.npz'
    
    folder = Path(folder)
    if folder.exists() == False:
        os.makedirs(str(folder.absolute()))

    filepath = folder.joinpath(filename)
    if filepath.exists():
        if flush:
            os.remove(str(filepath.absolute()))
        else:
            return np.load(str(filepath.absolute()))
        
    svx = compute_supervoxel(seg, seg_res, seg_origin, svx_coord, svx_size, vx_res, ct_df)
    np.savez(filepath, svx)

    return svx


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to folder to store the downloaded supervoxel')
    parser.add_argument('svx_x', type=int)
    parser.add_argument('svx_y', type=int)
    parser.add_argument('svx_z', type=int)
    parser.add_argument('-s', '--silent', action='store_true')

    args = parser.parse_args()

    dataset_name = 'minnie65_public'
    materialization = 1300      # Required for mesh downloading
    
    client = CAVEclient(dataset_name)
    client.version = materialization

    seg_cv = CloudVolume('precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg_m1300', progress=False, use_https=True)
    img_cv = CloudVolume('precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/em', progress=True, use_https=True)

    meta = client.materialize.get_table_metadata("allen_v1_column_types_slanted_ref")
    ann_res = Vec(*meta["voxel_resolution"])
    seg_res = seg_cv.resolution

    ct_all_df = get_all_cts(client)

    ###############################
    # NOTE: MUST BE EDGE ALIGNED!!!
    ###############################

    dx_in = int(seg_res[0])
    dx_out = 100

    dz_in = int(seg_res[2])
    dz_out = 100

    n_lat = lcm(dx_in, dx_out) / dx_in
    n_depth = lcm(dz_in, dz_out) / dz_in

    # TODO: Update to be edge aligned (only memory-based sizing part)
    # arr_size = 4 * pow(1024, 3)
    # lateral_vx = np.cbrt(5 * (arr_size / 8))
    # depth_vx = lateral_vx / 5
    
    depth_vx = 8 * n_depth                  # vx
    lateral_vx = 50 * n_lat                 # vx

    print(f'Estimated array size: {(lateral_vx * lateral_vx * depth_vx * 8) / pow(1024, 3) :.2f} GB')

    depth_nm = depth_vx * seg_res[2]
    lateral_nm = lateral_vx * seg_res[0]

    origin = Vec(681280, 531968, 809000)    # nm
    origin = origin.astype(int)             # nm

    svx_coord = (args.svx_x, args.svx_y, args.svx_z)
    svx = memoize_supervoxel(args.path, seg_cv, seg_res, origin, svx_coord, (lateral_nm, lateral_nm, depth_nm), (100, 100, 100), ct_all_df)

    if not args.silent:
        import matplotlib.pyplot as plt
        plt.imshow(svx[:,:,0], cmap='gray')
        plt.show()
