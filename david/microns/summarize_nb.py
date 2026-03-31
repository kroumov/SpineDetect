import json
import sys

def summarize_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print(f"Summary of {path}:")
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            lines = cell['source']
            if lines:
                # Print first non-empty/non-comment line if possible, or just the first line
                content = "".join(lines).strip()
                # summarized =  content.split('\n')[0]
                # print(f"Cell {i} [Code]: {summarized[:100]}...")
                print(f"Cell {i} [Code]:")
                print(content[:200] + "..." if len(content) > 200 else content)
                print("-" * 20)
        elif cell['cell_type'] == 'markdown':
            lines = cell['source']
            if lines:
                content = "".join(lines).strip()
                print(f"Cell {i} [Markdown]: {content[:100]}...")
                print("-" * 20)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        summarize_notebook(sys.argv[1])
