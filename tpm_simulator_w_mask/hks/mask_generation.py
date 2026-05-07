import os
from pathlib import Path

import numpy as np
import pandas as pd
from caveclient import CAVEclient
from cloudvolume import CloudVolume
import trimesh  # Using meshparty for the base mesh
from meshparty import trimesh_io

from meshmash.pipeline import chunked_hks_pipeline

from scipy import ndimage
from skimage.transform import resize

import joblib
from collections import Counter

import tifffile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DIR = Path(__file__).resolve().parent
DOWNLOAD_BASE = PROJECT_ROOT / "data" / "download"

def get_hks(
    root_ids, mesh_dict, client, distance_threshold=1000, 
    mapping_column="ctr_pt_position", side="post"
):
    results = {}
    for root_id in root_ids:
        print(f"\nProcessing root_id: {root_id}")
        if root_id not in mesh_dict:
            continue
        
        mesh = mesh_dict[root_id]
        print(f"  Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
        
        # Extract raw arrays for Cython
        raw_vertices = np.array(mesh.vertices, dtype=np.float32)
        raw_faces = np.array(mesh.faces, dtype=np.int64)
        
        # 2. BYPASS THE BUG: Set query_indices=None 
        hks_result = chunked_hks_pipeline(
            mesh=(raw_vertices, raw_faces), 
            query_indices=None,  # <--- This prevents the crash!
            simplify_agg=7,
            simplify_target_reduction=0.7, 
            overlap_distance=20_000,
            verbose=True,
        )
        
        results[root_id] = {
            'hks_result': hks_result
        }
        print(f"  Success! HKS computed for root_id {root_id}.")
        
    return results

def get_mesh(
    root_ids, client
):
    # --- Execution Block ---
    client.version = 1300

    mesh_dict = {}

    cv_path = client.info.segmentation_source()
    cv = CloudVolume(cv_path, use_https=True, progress=True)

    print(f"\nBatch fetching meshes for: {root_ids}")

    try:
        # 1. BATCH DOWNLOAD (Fast parallel fetching)
        batch_mesh_data = cv.mesh.get(root_ids, fuse=False)
        
        # 2. Iterate through the downloaded batch
        for root_id, mesh_data in batch_mesh_data.items():
            print(f"\nProcessing downloaded mesh for {root_id}...")
            
            if isinstance(mesh_data, dict):
                all_vertices, all_faces, vertex_offset = [], [], 0
                for chunk_id, sub_mesh in mesh_data.items():
                    v = sub_mesh.vertices if hasattr(sub_mesh, 'vertices') else sub_mesh['vertices']
                    f = sub_mesh.faces if hasattr(sub_mesh, 'faces') else sub_mesh['faces']
                    
                    all_vertices.append(np.array(v))
                    all_faces.append(np.array(f).reshape(-1, 3) + vertex_offset) 
                    vertex_offset += len(v)
                    
                final_vertices = np.vstack(all_vertices)
                final_faces = np.vstack(all_faces)
                
                # --- THE HEALING CHAMBER ---
                print("  Healing mesh seams...")
                temp_mesh = trimesh.Trimesh(vertices=final_vertices, faces=final_faces, process=True)
                
                # Extract and cast explicitly
                clean_v = np.array(temp_mesh.vertices, dtype=np.float32)
                clean_f = np.array(temp_mesh.faces, dtype=np.int64)
                
                # Lock into meshparty safely
                m = trimesh_io.Mesh(vertices=clean_v, faces=clean_f, process=False)
                
            elif hasattr(mesh_data, 'vertices'):
                print("  Healing standard mesh...")
                temp_mesh = trimesh.Trimesh(
                    vertices=np.array(mesh_data.vertices), 
                    faces=np.array(mesh_data.faces).reshape(-1, 3), 
                    process=True
                )
                clean_v = np.array(temp_mesh.vertices, dtype=np.float32)
                clean_f = np.array(temp_mesh.faces, dtype=np.int64)
                m = trimesh_io.Mesh(vertices=clean_v, faces=clean_f, process=False)
                
            mesh_dict[root_id] = m
            print(f"  Successfully loaded locked & clean mesh: {len(m.vertices)} vertices")
            
    except Exception as e:
        print(f"Batch processing failed: {e}")

    return mesh_dict

def generate_tags(hks_features, wd):
    ensemble_models = joblib.load(os.path.join(wd, r'rf_ensemble.pkl'))

    preds = np.array([model.predict(hks_features) for model in ensemble_models])
    # shape: (n_models, n_samples)

    tag_pred = np.array([
        Counter(preds[:, i]).most_common(1)[0][0]
        for i in range(preds.shape[1])
    ])
    return tag_pred

def generate_submesh(vertex_mask, vertices, faces):
        # 1. select faces
        face_mask = vertex_mask[faces].all(axis=1)
        selected_faces = faces[face_mask]

        # 2. build submesh
        unique_vertices, new_indices = np.unique(selected_faces, return_inverse=True)
        new_vertices = vertices[unique_vertices]
        new_faces = new_indices.reshape(-1, 3)

        submesh = trimesh.Trimesh(new_vertices, new_faces, process=False)
        
        return submesh

def mesh2mask(mesh, bbox_min, pitch, grid_size):
    try:
        vox = mesh.voxelized(pitch=pitch)
        vox = vox.fill()
    except Exception as e:
        print(f"Error occurred while voxelizing submesh: {e}")
        return None

    points = vox.points   # coordinates of filled voxels
    indices = ((points - bbox_min) / pitch).astype(int)

    binary_mask = np.zeros(grid_size + 1, dtype=np.uint8)

    binary_mask[
        indices[:, 0],
        indices[:, 1],
        indices[:, 2]
    ] = 1

    filled_mask = binary_mask # ndimage.binary_fill_holes(binary_mask)
    for x in range(binary_mask.shape[0]):
        filled_mask[x, :, :] = ndimage.binary_fill_holes(filled_mask[x, :, :])
    for y in range(binary_mask.shape[1]):
        filled_mask[:, y, :] = ndimage.binary_fill_holes(filled_mask[:, y, :])
    for z in range(binary_mask.shape[2]):
        filled_mask[:, :, z] = ndimage.binary_fill_holes(filled_mask[:, :, z])

    filled_mask = ndimage.binary_closing(filled_mask, structure=np.ones((3,3,3)))
    # filled_mask = ndimage.binary_opening(filled_mask, structure=np.ones((3,3,3)))
    
    return filled_mask

def generate_masks(simple_vertices, simple_faces, tag_pred, bbox_min, bbox_max, pitch):
    extent = bbox_max - bbox_min

    grid_size = np.ceil(extent / pitch).astype(int)

    masks = np.zeros((*(grid_size + 1), len(np.unique(tag_pred))), dtype=np.uint8)  # +1 to include the max edge

    t_count = 0

    for t in np.unique(tag_pred):
        # 1. mask vertices
        vertex_mask = (tag_pred == t) & ((simple_vertices >= bbox_min) & (simple_vertices <= bbox_max)).all(axis=1)  # Example: only look at the specified block for this tag

        # 2. generate submesh
        submesh = generate_submesh(vertex_mask, simple_vertices, simple_faces)

        # 3. voxelize + fill
        filled_mask = mesh2mask(submesh, bbox_min, pitch, grid_size)
        if filled_mask is None:
            continue

        masks[:, :, :, t_count] = filled_mask
        t_count += 1

    return masks
    
def ensure_mutual_exclusion(masks):
    # ensure masks are mutually exclusive
    masks[:,:,:,1] = masks[:,:,:,1] & (~masks[:,:,:,0])
    masks[:,:,:,2] = masks[:,:,:,2] & (~masks[:,:,:,0]) & (~masks[:,:,:,1])
    
    return masks

def save_masks_as_tiff(masks, identifier, output_dir=DOWNLOAD_BASE):
    masks = masks.astype(np.uint8)  # Ensure data is uint8 for TIFF
    masks_tiff = np.transpose(masks, (2, 3, 1, 0))  # Move tag dimension to the front
    filename = f"masks_{identifier}.tiff"
    folder = f"microns_{identifier}"
    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    tifffile.imwrite(os.path.join(output_dir, folder, filename), masks_tiff, photometric='minisblack', metadata={'axes': 'ZCYX'}, imagej=True)


def mask_generation_pipeline(
        identifier,
        root_ids, 
        client, 
        bbox_min, 
        bbox_max, 
        res_seg,
        wd = LOCAL_DIR):
    mesh_dict = get_mesh(root_ids, client)
    hks_results = get_hks(root_ids, mesh_dict, client)

    masks = None
    pitch = np.min(res_seg)
    for root_id in root_ids:
        if root_id not in hks_results:
            continue
        
        hks_features = hks_results[root_id]['hks_result'].simple_features
        tag_pred = generate_tags(hks_features, wd=wd)

        simple_vertices, simple_faces = hks_results[root_id]['hks_result'].simple_mesh

        if masks is None:
            masks = generate_masks(simple_vertices, simple_faces, tag_pred, bbox_min, bbox_max, pitch)
        else:
            new_masks = generate_masks(simple_vertices, simple_faces, tag_pred, bbox_min, bbox_max, pitch)
            if new_masks is not None:
                masks = masks | new_masks  # Combine masks from different root_ids

    # resize mask to match segmentation resolution
    resized_masks = None
    target_shape = (bbox_max - bbox_min) / res_seg

    for ch in range(masks.shape[3]):
        mask = masks[:, :, :, ch]
        resized_mask = resize(mask, target_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
        if resized_masks is None:
          resized_masks = np.zeros((* resized_mask.shape, masks.shape[3]), dtype=np.uint8)
        resized_masks[:, :, :, ch] = resized_mask

    save_masks_as_tiff(resized_masks, identifier, output_dir=DOWNLOAD_BASE)

if __name__ == "__main__":
    # Example usage
    root_ids = [864691135645549807]  # Replace with actual root IDs [864691135995447722, 864691136903982002, 864691135698196757, 864691136451925247, 864691135841778147, 864691136310935130, 864691135995909546, 864691137054525558, 864691135590799371, 864691135698193941, 864691135133519520, 864691135915774822, 864691135856767534]
    client = CAVEclient('minnie65_public')  # Initialize your client here

    identifier = root_ids[0]  # Use the first root_id as identifier for output files
    bbox_min = np.array([1060288.0, 567936.0, 730880.0])  # Replace with actual bounding box min [32768, 32768, 81920]
    bbox_max = np.array([32.768, 32.768, 81.92]) * 1000 + bbox_min  # Replace with actual bounding box max [991872.0, 485568.0, 597680.0] + [32768, 32768, 81920]
    res_seg = np.array([64.0, 64.0, 80.0])  # Voxel size in nm

    mask_generation_pipeline(identifier, root_ids, client, bbox_min, bbox_max, res_seg)