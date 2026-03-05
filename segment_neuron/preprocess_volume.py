import os
import sys
import numpy as np
import tifffile
from pathlib import Path
from scipy import ndimage as ndi
import time

def run_preprocessing(input_path, output_path):
    print(f"[{time.strftime('%H:%M:%S')}] Initializing volume loading: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    data = tifffile.imread(input_path)
    print(f"   - Volume dimensions: {data.shape}, dtype: {data.dtype}")
    
    # Phase 1: Range Normalization
    print(f"[{time.strftime('%H:%M:%S')}] Normalizing intensity range...")
    d_min, d_max = data.min(), data.max()
    if d_max > d_min:
        data = (data - d_min) / (d_max - d_min)
    
    # Phase 2: 3D Morphological Repair
    print(f"[{time.strftime('%H:%M:%S')}] Executing 3D morphological closing...")
    struct = ndi.generate_binary_structure(3, 1)
    # Perform thresholding and closing in a single step to save memory
    binary = data > 0.5
    del data
    
    processed = ndi.binary_closing(binary, structure=struct)
    del binary
    
    # Phase 3: Export Clean Pixels
    print(f"[{time.strftime('%H:%M:%S')}] Exporting processed volume: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export as raw uint8 binary map
    tifffile.imwrite(output_path, (processed.astype(np.uint8) * 255))
    
    print(f"\n[SUCCESS] Preprocessing pipeline complete.")

if __name__ == "__main__":
    # Resolve relative paths
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    
    input_file = root_dir / "output.tiff"
    output_file = script_dir / "results" / "preprocessed_output.tiff"
    
    run_preprocessing(str(input_file), output_file)
