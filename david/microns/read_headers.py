import json
import sys

def read_headers(path):
    print(f"--- Headers for {path} ---")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        for cell in nb['cells']:
            if cell['cell_type'] == 'markdown':
                content = ''.join(cell['source']).strip()
                if content.startswith('#'):
                    # Print only lines starting with # to verify structure
                    for line in content.split('\n'):
                        if line.strip().startswith('#'):
                            print(line.strip())
    except Exception as e:
        print(f"Error reading {path}: {e}")
    print("\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            read_headers(path)
