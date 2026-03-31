import os
import sys
import numpy as np
import tifffile
from pathlib import Path
from skimage import measure, segmentation
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

def run_segmentation(input_path, seed_path, output_dir):
    print(f"[{time.strftime('%H:%M:%S')}] Loading volume data and building 3D topography...")
    binary = tifffile.imread(input_path) > 0
    seeds = tifffile.imread(seed_path).astype(np.int32)
    
    # Generate Topography (Inverse Distance Field)
    # The skeleton (centerline) becomes the valley bottom
    dt = ndi.distance_transform_edt(binary)
    topo = -dt
    
    # Phase 1: 3D Seeded Watershed
    print(f"[{time.strftime('%H:%M:%S')}] Executing 3D seeded watershed segmentation...")
    labels = segmentation.watershed(topo, seeds, mask=binary)
    
    # Phase 2: Export Data
    print(f"[{time.strftime('%H:%M:%S')}] Saving label maps and RGB volume...")
    output_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_dir / "segmented_volume.tiff", labels.astype(np.uint8))
    
    # RGB Mapping for Visualization (Fiji Compatible)
    z, y, x = labels.shape
    rgb = np.zeros((z, y, x, 3), dtype=np.uint8)
    rgb[labels == 1] = [0, 255, 255] # Trunk -> Cyan
    rgb[labels == 2] = [255, 136, 0] # Branches -> Orange
    
    tifffile.imwrite(
        output_dir / "colored_segmented_volume.tiff", 
        rgb, 
        photometric='rgb',
        metadata={'axes': 'ZYXC'},
        compression='zlib'
    )
    del rgb

    # Phase 3: High-Resolution 3D Preview
    print(f"[{time.strftime('%H:%M:%S')}] Generating high-fidelity 3D preview...")
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    # Full resolution surface reconstruction (Stride=1)
    stride = 1
    
    def render_mesh(mask, color, alpha):
        if mask.any():
            v, f, _, _ = measure.marching_cubes(mask[::stride, ::stride, ::stride], level=0.5)
            v_aligned = v * stride
            ax.add_collection3d(Poly3DCollection(v_aligned[f, :][:, :, [2, 1, 0]], facecolor=color, alpha=alpha, edgecolor='none'))

    render_mesh(labels == 1, '#00FFFF', 0.6)
    render_mesh(labels == 2, '#FF8800', 0.8)

    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect((binary.shape[2], binary.shape[1], binary.shape[0]))
    ax.set_title("3D Segmented Neuronal Volume", color='black', fontsize=20, pad=40)
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlim(0, binary.shape[2]); ax.set_ylim(0, binary.shape[1]); ax.set_zlim(0, binary.shape[0])

    plt.savefig(output_dir / "segmented_volume_3d.png", dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\n[SUCCESS] Segmentation pipeline complete.")
    print(f"   - Label map: {output_dir / 'segmented_volume.tiff'}")
    print(f"   - RGB Volume: {output_dir / 'colored_segmented_volume.tiff'}")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    bin_file = script_dir / "results" / "preprocessed_output.tiff"
    seed_file = script_dir / "results" / "skeleton_seeds.tiff"
    results_dir = script_dir / "results"
    
    if not bin_file.exists() or not seed_file.exists():
        print("Error: Required input files missing. Ensure preprocessing and skeletonization are completed.")
        sys.exit(1)
        
    run_segmentation(str(bin_file), str(seed_file), results_dir)
