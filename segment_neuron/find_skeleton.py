import os
import sys
import numpy as np
import tifffile
from pathlib import Path
from skimage import measure, morphology
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import pandas as pd
import time

def draw_volumetric_marker(ax, center, radius, color='#FF0000', alpha=0.1):
    """Render a 3D parametric sphere representing local volume."""
    if radius < 0.5: return 
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 10)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, antialiased=False, zorder=5)

def analyze_topology(input_path, output_dir):
    print(f"[{time.strftime('%H:%M:%S')}] Executing skeletonization and segment extraction...")
    data = tifffile.imread(input_path)
    binary = data > 0
    skeleton = morphology.skeletonize(binary)
    skel_coords = np.argwhere(skeleton)
    skel_set = set(map(tuple, skel_coords))
    dt = ndi.distance_transform_edt(binary)
    
    # Identify Topological Nodes (Terminals & Junctions)
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0
    neighbors = ndi.convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    terminals = set(tuple(c) for c in np.argwhere((neighbors == 1) & skeleton))
    junctions = set(tuple(c) for c in np.argwhere((neighbors >= 3) & skeleton))
    nodes = terminals | junctions
    
    # Path Segment Extraction
    segments_terminal = [] # Ends at a terminal point
    segments_internal = [] # Connects two junction points
    
    visited = set()
    offsets = [(dz, dy, dx) for dz in [-1,0,1] for dy in [-1,0,1] for dx in [-1,0,1] if not (dz==dy==dx==0)]
    
    for start in nodes:
        for dz, dy, dx in offsets:
            adj = (start[0]+dz, start[1]+dy, start[2]+dx)
            if adj in skel_set:
                edge_id = tuple(sorted([start, adj]))
                if edge_id in visited: continue
                
                path = [start, adj]
                curr, prev = adj, start
                while curr not in nodes:
                    next_p = None
                    for ndz, ndy, ndx in offsets:
                        cand = (curr[0]+ndz, curr[1]+ndy, curr[2]+ndx)
                        if cand in skel_set and cand != prev:
                            next_p = cand
                            break
                    if next_p:
                        path.append(next_p)
                        prev, curr = curr, next_p
                    else: break
                
                visited.add(tuple(sorted([start, adj])))
                visited.add(tuple(sorted([path[-1], path[-2]]))) 
                
                if start in terminals or curr in terminals:
                    segments_terminal.append(path)
                else:
                    segments_internal.append(path)

    # Classification via Length-Knee Analysis
    print(f"[{time.strftime('%H:%M:%S')}] Classifying branches and main trunks...")
    
    def calc_len(p):
        return sum(np.linalg.norm(np.array(p[i]) - np.array(p[i+1])) for i in range(len(p)-1))

    lengths = [calc_len(p) for p in segments_terminal]
    final_branches, final_trunks = [], segments_internal.copy()
    
    if len(lengths) > 2:
        s_idx = np.argsort(lengths)
        s_len = np.array(lengths)[s_idx]
        x_n = np.linspace(0, 1, len(s_len))
        y_n = (s_len - s_len.min()) / (s_len.max() - s_len.min())
        knee = np.argmin(y_n - x_n)
        thresh = s_len[knee]
        
        for i, p in enumerate(segments_terminal):
            if lengths[i] > thresh: final_trunks.append(p)
            else: final_branches.append(p)
    else:
        final_branches = segments_terminal

    # Export Visualization
    print(f"[{time.strftime('%H:%M:%S')}] Rendering 3D results...")
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    stride = 3
    v, f, _, _ = measure.marching_cubes(binary[::stride, ::stride, ::stride], level=0.5)
    v_aligned = v[:, [2, 1, 0]] * stride
    ax.add_collection3d(Poly3DCollection(v_aligned[f], alpha=0.05, facecolor='#CCCCCC', edgecolor='none'))

    def plot_paths(paths, color, lw, alpha, z):
        segs = []
        for p in paths:
            for i in range(len(p)-1):
                segs.append([(p[i][2], p[i][1], p[i][0]), (p[i+1][2], p[i+1][1], p[i+1][0])])
        if segs:
            ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=lw, alpha=alpha, zorder=z))

    plot_paths(final_trunks, '#00FFFF', 0.8, 0.6, 0)
    plot_paths(final_branches, '#FF8800', 1.2, 0.9, 10)

    for z, y, x in junctions:
        draw_volumetric_marker(ax, (x, y, z), dt[z,y,x], '#FF0000', 0.05)
        ax.scatter(x, y, z, c='#FF0000', s=0.5, alpha=1.0, edgecolors='black', linewidth=0.2, zorder=20)
    for z, y, x in terminals:
        draw_volumetric_marker(ax, (x, y, z), dt[z,y,x], '#00FF00', 0.05)
        ax.scatter(x, y, z, c='#00FF00', s=0.5, alpha=1.0, edgecolors='black', linewidth=0.2, zorder=20)

    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect((data.shape[2], data.shape[1], data.shape[0]))
    ax.set_title("3D Neuronal Skeleton Analysis", color='black', fontsize=20, pad=40)
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlim(0, data.shape[2]); ax.set_ylim(0, data.shape[1]); ax.set_zlim(0, data.shape[0])

    plt.savefig(output_dir / "connectivity_branches.png", dpi=300, bbox_inches='tight', facecolor='white')
    
    # Save Outputs
    seed_vol = np.zeros(data.shape, dtype=np.uint16)
    for p in final_trunks:
        for pt in p: seed_vol[pt] = 1
    for p in final_branches:
        for pt in p: seed_vol[pt] = 2
    tifffile.imwrite(output_dir / "skeleton_seeds.tiff", seed_vol)
    
    nodes_df = [{'z':z, 'y':y, 'x':x, 'r':dt[z,y,x], 'type':'junction'} for z,y,x in junctions] + \
               [{'z':z, 'y':y, 'x':x, 'r':dt[z,y,x], 'type':'terminal'} for z,y,x in terminals]
    pd.DataFrame(nodes_df).to_csv(output_dir / "skeleton_nodes.csv", index=False)

    print(f"\n[SUCCESS] Skeleton analysis completed. Outputs saved to {output_dir}")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    input_file = script_dir / "results" / "preprocessed_output.tiff"
    output_dir = script_dir / "results"
    analyze_topology(str(input_file), output_dir)
