import pandas as pd
import numpy as np
import ast
import scipy.spatial
from caveclient import CAVEclient
from meshparty import trimesh_io
from meshmash import condensed_hks_pipeline
import cloudvolume
import trimesh  # Using meshparty for the base mesh
from meshmash.cave import get_synapse_mapping
from meshmash.pipeline import chunked_hks_pipeline
from meshmash.utils import project_points_to_mesh

import annotation_utils
import importlib
importlib.reload(annotation_utils)
from multiprocessing import Pool


def get_mesh_dict(test_root_ids):
    try:
        # 1. BATCH DOWNLOAD (Fast parallel fetching)
        batch_mesh_data = cv.mesh.get(test_root_ids, fuse=False)
        
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
        return mesh_dict  
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return None



    # Run pipeline
    results = get_hks_for_synapses(vortex_df, test_root_ids, mesh_dict, client)

    for root_id, data in results.items():
        hks_features = data['hks_result'].simple_features
        print(f"Root {root_id}: {len(hks_features)} HKS features computed")

    return mesh_dict

def get_hks_for_synapses(
    synapse_df, root_ids, mesh_dict, client, distance_threshold=1000, 
    mapping_column="ctr_pt_position", side="post"
):
    results = {}
    for root_id in root_ids:
        print(f"\nProcessing root_id: {root_id}")
        if root_id not in mesh_dict:
            continue
        
        mesh = mesh_dict[root_id]
        
        # 1. Map synapses to the original, full-resolution mesh
        synapse_mesh_mapping = get_synapse_mapping(
            root_id, mesh, client,
            distance_threshold=distance_threshold,
            mapping_column=mapping_column,
            side=side,
        )
        
        if len(synapse_mesh_mapping) == 0:
            print(f"  No synapses mapped.")
            continue
        
        original_synapse_indices = synapse_mesh_mapping[:, 1].astype(np.int64)
        print(f"  Mapped {len(original_synapse_indices)} synapses. Running HKS pipeline...")
        
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
        
        # 3. LOOK UP THE FEATURES AFTERWARDS
        # hks_result.mapping translates original indices to simplified indices
        simplified_synapse_indices = hks_result.mapping[original_synapse_indices]
        
        # Drop any synapses that were completely deleted during simplification (-1)
        valid_mask = simplified_synapse_indices != -1
        valid_simplified_indices = simplified_synapse_indices[valid_mask]
        
        # Extract just the features for our synapses
        # hks_result.simple_features contains the computed data
        synapse_features = hks_result.simple_features.iloc[valid_simplified_indices]
        
        results[root_id] = {
            'synapse_mapping': synapse_mesh_mapping[valid_mask], 
            'hks_result': hks_result,
            'synapse_features': synapse_features
        }
        print(f"  Success! Extracted features for {len(valid_simplified_indices)} synapses.")
        
    return results



if __name__ == "__main__":
    client = CAVEclient('minnie65_public')
    cv = cloudvolume.CloudVolume(client.info.segmentation_source(), use_https=True)
    client.materialize.get_tables()
    
    # Get vortex data
    vortex_df = client.materialize.query_table(
        'vortex_compartment_targets',
        filter_equal_dict={'valid': True}
    )

    # 1. Pull the data from the vortex_compartment_targets table
    # (You might want to remove the limit=1000 if you want the whole dataset!)
    vortex_df = client.materialize.query_table(
        'vortex_compartment_targets',
        filter_equal_dict={'valid': True}
    )

    # 2. Keep only the columns your ML model cares about
    vortex_df = vortex_df[['id', 'post_pt_root_id', 'tag', 'post_pt_position']]

    # 3. Find the neurons with the LEAST connections
    counts = vortex_df['post_pt_root_id'].value_counts(ascending=True)

    # Let's grab the 23 neurons with the fewest synapses for your test
    test_root_ids = counts.iloc[-30:-7].index.to_list()

    client.version = 1300

    mesh_dict = {}

    cv_path = client.info.segmentation_source()
    cv = cloudvolume.CloudVolume(cv_path, use_https=True, progress=True)

    mesh_dict = get_mesh_dict(test_root_ids)

    with Pool(processes=5) as pool:
        results = pool.map(get_hks_for_synapses, [(vortex_df, root_id, mesh_dict, client) for root_id in test_root_ids])

    all_ml_data = []

    # CAVE usually returns points in voxels. Meshes are in nanometers.
    # This multiplier converts voxel coordinates to nanometers.
    # (If your dataset uses a different resolution, adjust this!)
    RESOLUTION = np.array([64, 64, 320]) 

    for root_id, data in results.items():
        
        # 1. Filter your ground truth dataframe for just this neuron
        neuron_df = vortex_df[vortex_df['post_pt_root_id'] == root_id].copy()
        
        if len(neuron_df) == 0:
            continue
            
        # 2. Extract the 3D coordinates of your manual annotations
        # Stack them into a 2D array and convert to nanometers
        pts = np.vstack(neuron_df['post_pt_position'].values)
        
        # Quick heuristic check: if coordinates are tiny, they need voxel-to-nm conversion
        if pts[0][0] < 1000000: 
            pts = pts * RESOLUTION
            
        # 3. Get the original mesh we stored in memory earlier
        original_mesh = mesh_dict[root_id]
        original_vertices = np.array(original_mesh.vertices)
        
        # 4. Use a KDTree to find the closest mesh vertex to each manual annotation
        tree = cKDTree(original_vertices)
        distances, original_vertex_indices = tree.query(pts)
        
        # 5. Translate that original vertex to the simplified mesh vertex
        hks_result = data['hks_result']
        mapping_array = hks_result.mapping
        simplified_vertex_indices = mapping_array[original_vertex_indices]
        
        # Filter out any points that landed on geometry that was deleted during simplification (-1)
        valid_mask = simplified_vertex_indices != -1
        valid_neuron_df = neuron_df[valid_mask].reset_index(drop=True)
        valid_simplified_indices = simplified_vertex_indices[valid_mask]
        
        # 6. Extract the 32 computed HKS features for these specific vertices
        hks_features = hks_result.simple_features.iloc[valid_simplified_indices].reset_index(drop=True)
        
        # 7. Glue the labels (tag) side-by-side with the math (HKS)
        combined_df = pd.concat([valid_neuron_df, hks_features], axis=1)
        
        all_ml_data.append(combined_df)

    # 8. Create the final, master ML dataset
    if len(all_ml_data) > 0:
        ml_ready_df = pd.concat(all_ml_data, ignore_index=True)
        print(f"Successfully built ML dataset with {len(ml_ready_df)} annotated targets.")
        print(ml_ready_df.shape)
    else:
        print("No data matched! Make sure vortex_df contains the neurons you processed.")

