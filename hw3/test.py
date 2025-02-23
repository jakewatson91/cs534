import numpy as np
from collections import deque
import json

def parse_files(file):
    with open(file, 'r') as f:
        data = json.load(f)
        train = data['train']
        test = data['test']
    return train, test

def bfs_all_combinations(grid, target):
    iterations = 0
    # Start with the original grid and an empty move list.
    q = deque([(grid, [])])
    visited = set()
    visited.add(tuple(map(tuple, grid)))
    
    while q:
        frontier, cur_mappings = q.popleft()
        # If we have reached the target, return the move list.
        if np.array_equal(frontier, target):
            print(f"Match found after {iterations} iterations!")
            return cur_mappings

        # Generate all possible moves that would correct one cell.
        mappings = generate_mapping(frontier, target)
        for mapping in mappings:
            new_grid = generate_output(frontier, mapping)
            new_grid_tuple = tuple(map(tuple, new_grid))
            if new_grid_tuple not in visited:
                visited.add(new_grid_tuple)
                q.append((new_grid, cur_mappings + [mapping]))
            iterations += 1

    print("No match found.")
    return None

def generate_mapping(frontier, target):
    """
    Generate a complete one-to-one mapping from every cell in the frontier grid
    to a unique cell in the target grid, where the values match.
    Assumes that for every value, the count in frontier equals the count in target.
    Returns a list of mappings in the form:
      (source_value, (source_row, source_col), target_value, (target_row, target_col))
    """
    res = []
    rows, cols = len(frontier), len(frontier[0])
    used_target = set()  # to track target cells already mapped
    
    # Iterate over every cell in the frontier grid
    for src_r in range(rows):
        for src_c in range(cols):
            src_val = frontier[src_r][src_c]
            found = False
            # Search for a target cell that has the same value and hasn't been used
            for tgt_r in range(rows):
                for tgt_c in range(cols):
                    if (tgt_r, tgt_c) in used_target:
                        continue
                    if target[tgt_r][tgt_c] == src_val:
                        res.append((src_val, (src_r, src_c), src_val, (tgt_r, tgt_c)))
                        used_target.add((tgt_r, tgt_c))
                        found = True
                        break
                if found:
                    break
            if not found:
                raise ValueError(f"No valid mapping found for cell ({src_r},{src_c}) with value {src_val}")
    return res


def generate_output(frontier, mapping):
    """
    Applies a single mapping to the frontier grid.
    It copies the source cell's value to the specified target coordinates,
    returning a new grid.
    """
    rows, cols = len(frontier), len(frontier[0])
    new_grid = [row[:] for row in frontier]
    # mapping is (value, (src_r, src_c), value, (tgt_r, tgt_c))
    _, _, new_val, (tgt_r, tgt_c) = mapping
    new_grid[tgt_r][tgt_c] = new_val
    return new_grid

if __name__ == "__main__":
    files = ['data/data_0.json', 'data/data_1.json', 'data/data_2.json', 'data/data_3.json']
    for file in files:
        train, test = parse_files(file)
        # Here we assume that train is a list of dicts with 'input' and 'output' keys.
        for pair in train:
            grid = pair['input']
            target = pair['output']
            print(f"Original grid: {grid}")
            print(f"Target grid: {target}")
            mappings = bfs_all_combinations(grid, target)
            print("Mappings: ", mappings)