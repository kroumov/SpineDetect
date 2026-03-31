import json
import sys

try:
    with open(r'c:\Users\bkrou\SpineDetect\david\microns\microns-segmentation.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print("--- Notebook Summary ---")
    for i, cell in enumerate(nb['cells']):
        ctype = cell['cell_type']
        source_lines = cell.get('source', [])
        if not source_lines:
            print(f"Cell {i} [{ctype}]: (Empty)")
            continue
            
        first_line = source_lines[0].strip()
        if len(first_line) > 80:
            first_line = first_line[:77] + "..."
            
        print(f"Cell {i} [{ctype}]: {first_line}")
        
except Exception as e:
    print(f"Error: {e}")
