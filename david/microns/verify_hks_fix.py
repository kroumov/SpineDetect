import numpy as np
import pandas as pd
from meshmash.pipeline import chunked_hks_pipeline

def create_dummy_mesh():
    # Component 1: A "large" tetrahedron (4 vertices)
    v1 = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    f1 = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ], dtype=np.int64)

    # Component 2: A "small" isolated point/triangle (3 vertices)
    # This will be dropped if min_vertex_threshold > 3
    v2 = np.array([
        [10, 10, 10],
        [11, 10, 10],
        [10, 11, 10]
    ], dtype=np.float32)
    f2 = np.array([
        [0, 1, 2]
    ], dtype=np.int64)

    vertices = np.vstack([v1, v2])
    faces = np.vstack([f1, f2 + 4])
    
    return (vertices, faces)

def test_hks_fix():
    print("Creating dummy mesh...")
    mesh = create_dummy_mesh()
    # Vertices 0-3 are in the large component (kept)
    # Vertices 4-6 are in the small component (dropped)
    
    # Query one vertex in the large component and one in the small
    query_indices = np.array([0, 5], dtype=np.int32)
    
    print(f"Original mesh vertex count: {len(mesh[0])}")
    print(f"Query indices: {query_indices}")
    
    try:
        print("\nRunning chunked_hks_pipeline with thresholding that drops vertex 5...")
        # min_vertex_threshold=4 will drop the triangle (v4, v5, v6)
        # simplify_target_reduction=0.0 ensures no further reduction for this small test
        
        # We need to make sure max_vertex_threshold is small enough to trigger chunking if we wanted,
        # but here we just want to see if subset_apply fails.
        result = chunked_hks_pipeline(
            mesh,
            query_indices=query_indices,
            min_vertex_threshold=4,
            simplify_target_reduction=0.0, 
            overlap_distance=1.0,
            max_vertex_threshold=1000,
            verbose=True
        )
        
        print("\nPipeline finished successfully!")
        
        # Check mapping
        mapping = result.mapping
        print(f"Vertex mapping size: {len(mapping)}")
        print(f"Vertex 0 maps to: {mapping[0]}")
        print(f"Vertex 5 maps to: {mapping[5]}")
        
        # In our case, vertex 0 should map to vertex 0 in the reduced mesh (since it's the first kept)
        # Vertex 5 should map to -1 (dropped)
        
        assert mapping[0] != -1, "Vertex 0 should NOT be dropped"
        assert mapping[5] == -1, "Vertex 5 SHOULD be dropped"
        
        print("\nSUCCESS: No IndexError occurred, and mapping correctly reflects thresholding.")
        
    except IndexError as e:
        print(f"\nFAILURE: IndexError occurred in test loop: {e}")
        # Re-raise to see the traceback
        raise e
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    test_hks_fix()
