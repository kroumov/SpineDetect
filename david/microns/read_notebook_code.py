import json
import sys

def read_nb(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            print(f"--- Cell {i} (Code) ---")
            print(''.join(cell['source']))
            print("\n")
        elif cell['cell_type'] == 'markdown':
            print(f"--- Cell {i} (Markdown) ---")
            for line in cell['source']:
                if line.strip().startswith('#'):
                    print(line.strip())
            print("\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        read_nb(sys.argv[1])
