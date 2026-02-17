
import numpy as np
import scipy.spatial
import pandas as pd
import json
import urllib.parse

def generate_ng_link(root_id, target_url='https://spelunker.cave-explorer.org/'):
    """
    Generates a Neuroglancer link using ONLY public sources (no auth needed).
    Follows the official nglui ViewerState.add_*_layer() pattern from the docs.
    
    Args:
        root_id (int): The Root ID of the neuron to visualize.
        target_url (str): Base URL for the Neuroglancer viewer.
    
    Returns:
        str: A Neuroglancer URL ready to open in a browser.
    """
    from nglui.statebuilder import ViewerState
    
    viewer = (
        ViewerState(dimensions=[4, 4, 40])
        .add_image_layer(
            source='precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em',
            name='imagery',
        )
        .add_segmentation_layer(
            source='precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg_m1300',
            name='segmentation',
            segments=[root_id],
        )
    )
    return viewer.to_url(target_url=target_url, shorten=False)


def generate_ng_link_with_auth(root_id, client):
    """
    Generates a Neuroglancer link using the graphene source (requires browser login).
    This gives full mesh + dynamic segmentation but needs middleauth cookies.
    
    To authenticate: visit https://minnie.microns-daf.com/segmentation/api/v1/table/minnie65_public/info
    in your browser first, log in with Google, then the link will work.
    
    Args:
        root_id (int): The Root ID of the neuron to visualize.
        client: A CAVEclient instance.
    
    Returns:
        str: A Neuroglancer URL (requires browser authentication).
    """
    from nglui.statebuilder import ViewerState
    from nglui.statebuilder.ngl_components import AnnotationLayer
    
    viewer = ViewerState(infer_coordinates=True)
    viewer.add_layers_from_client(client, imagery=True, segmentation=True)
    viewer.add_segments(segments=[root_id])
    
    # Add annotation layer for labeling
    viewer.add_layer(
        AnnotationLayer(
            name='spine-labels',
            tags=['spine', 'shaft', 'soma'],
            linked_segmentation='segmentation',
        )
    )
    
    return viewer.to_url(shorten=False)

def extract_points_from_ng_state(state_json, layer_name='spine-labels'):
    """
    Parses a Neuroglancer JSON state to extract point coordinates and tags.
    
    Args:
        state_json (dict): The full JSON state from Neuroglancer.
        layer_name (str): The name of the annotation layer to parse.
        
    Returns:
        tuple: (points, labels)
            points: Nx3 np.array of coordinates (in nm).
            labels: Nx1 np.array of class labels (1=spine, 2=shaft, 3=soma, 0=other).
    """
    layers = state_json.get('layers', [])
    points = []
    labels = []
    
    # Map binary props array [is_spine, is_shaft, is_soma] to integer class
    # based on the user's JSON "annotationProperties" order: spine, shaft, soma
    # props: [1, 0, 0] -> spine (1)
    # props: [0, 1, 0] -> shaft (2)
    # props: [0, 0, 1] -> soma (3)
    
    for layer in layers:
        if layer['name'] == layer_name:
            annotations = layer.get('annotations', [])
            for ann in annotations:
                if ann['type'] == 'point':
                    # Extract Coordinate (Neuroglancer uses physical units in this state)
                    pt = ann['point']
                    points.append(pt)
                    
                    # Extract Label from 'props'
                    props = ann.get('props', [0, 0, 0])
                    if props[0] == 1:
                        labels.append(1) # Spine
                    elif props[1] == 1:
                        labels.append(2) # Shaft
                    elif props[2] == 1:
                        labels.append(3) # Soma
                    else:
                        labels.append(0) # Unknown/Other
                    
    return np.array(points), np.array(labels)

def snap_points_to_mesh(mesh_vertices, points):
    """
    Finds the nearest mesh vertex index for each point.
    
    Args:
        mesh_vertices (np.array): Mx3 array of mesh vertex coordinates.
        points (np.array): Nx3 array of query points.
        
    Returns:
        tuple: (distances, vertex_indices)
            distances: Array of distances from point to nearest vertex.
            vertex_indices: Array of indices in mesh_vertices closest to each point.
    """
    if len(points) == 0:
        return np.array([]), np.array([])
        
    # Build K-D Tree for efficient nearest neighbor search
    tree = scipy.spatial.KDTree(mesh_vertices)
    
    # Query the tree
    distances, vertex_indices = tree.query(points)
    
    return distances, vertex_indices

def map_indices_to_segments(vertex_indices, segment_labels):
    """
    Maps mesh vertex indices to their corresponding Segment IDs (from HKS).
    
    Args:
        vertex_indices (np.array): Indices of vertices.
        segment_labels (np.array): Array where index i is the segment ID of vertex i.
        
    Returns:
        np.array: Array of Segment IDs corresponding to the vertices.
    """
    return segment_labels[vertex_indices]

def aggregate_k_nearest_hks(target_vertex_indices, mesh_vertices, hks_features, k=50):
    """
    Computes aggregated HKS features for each target vertex based on its k-nearest neighbors.
    
    Args:
        target_vertex_indices (np.array): Indices of the vertices to compute features for.
        mesh_vertices (np.array): Mx3 array of all mesh vertex coordinates.
        hks_features (pd.DataFrame): The DataFrame containing HKS features (index=vertex_index).
        k (int): Number of nearest neighbors to aggregate.
        
    Returns:
        pd.DataFrame: DataFrame where each row corresponds to a target vertex, containing aggregated features.
    """
    if len(target_vertex_indices) == 0:
        return pd.DataFrame()

    print(f"Building KD-Tree for {len(mesh_vertices)} vertices...")
    tree = scipy.spatial.KDTree(mesh_vertices)
    
    # Get coordinates of target vertices
    target_coords = mesh_vertices[target_vertex_indices]
    
    # Find k nearest neighbors for all targets at once
    print(f"Querying k={k} nearest neighbors for {len(target_coords)} points...")
    _, neighbor_indices = tree.query(target_coords, k=k)
    
    aggregated_data = []
    
    for i, neighbors in enumerate(neighbor_indices):
        # Extract HKS features for these neighbors
        # hks_features index is assumed to be vertex index
        neighbor_feats = hks_features.loc[neighbors]
        
        # Compute statistics across the k neighbors for each HKS column (time scale)
        # Result is a Series with MultiIndex (column, stat) or similar
        stats = neighbor_feats.agg(['mean', 'std', 'min', 'max', 'median'])
        
        # Flatten into a single feature vector
        # e.g. hks_t1_mean, hks_t1_std, ...
        row = {}
        row['vertex_index'] = target_vertex_indices[i]
        
        for col in stats.columns:
            for stat_name in stats.index:
                row[f'{col}_{stat_name}'] = stats.loc[stat_name, col]
                
        aggregated_data.append(row)
        
    return pd.DataFrame(aggregated_data)

def create_training_dataset(target_vertex_indices, labels, mesh_vertices, hks_features, k=50):
    """
    Creates the X (features) and y (labels) for ML training using KNN aggregation.
    
    Args:
        target_vertex_indices (np.array): Indices of the snapped vertices.
        labels (np.array): Class labels corresponding to each vertex (1=spine, 2=shaft, etc).
        mesh_vertices (np.array): All mesh vertices.
        hks_features (pd.DataFrame): HKS features for all vertices.
        k (int): Number of neighbors.
        
    Returns:
        tuple: (X, y) as DataFrames/Series ready for sklearn.
    """
    # 1. Aggregate Features
    print("Aggregating features from neighbors...")
    X = aggregate_k_nearest_hks(target_vertex_indices, mesh_vertices, hks_features, k=k)
    
    # 2. Prepare Labels
    # Ensure labels match the order of X (which comes from target_vertex_indices)
    y = pd.Series(labels, name='label')
    
    return X, y
